"""
Registry system inspired by MMSeg/MMEngine.
Allows registering and building modules by name from config dicts.
"""

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """A registry to map strings to classes or functions.

    Similar to MMEngine's Registry, this allows building modules from
    config dicts like: {'type': 'ResNet', 'depth': 50, 'pretrained': True}

    Example:
        >>> BACKBONES = Registry('backbone')
        >>> @BACKBONES.register_module()
        ... class ResNet:
        ...     pass
        >>> backbone = BACKBONES.build({'type': 'ResNet', 'depth': 50})
    """

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> Dict[str, Any]:
        return self._module_dict

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self._name}, "
            f"items={list(self._module_dict.keys())})"
        )

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    def get(self, key: str) -> Optional[Any]:
        return self._module_dict.get(key)

    def register_module(
        self,
        name: Optional[str] = None,
        force: bool = False,
        module: Optional[Any] = None,
    ) -> Callable:
        """Register a module.

        Args:
            name: Module name. Defaults to class/function name.
            force: Whether to override existing module with same name.
            module: Module class/function to be registered directly.

        Returns:
            Decorator or the module itself if passed directly.
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a bool, but got {type(force)}")

        # Called as: @registry.register_module() or registry.register_module(module=Cls)
        if module is not None:
            self._do_register(name or module.__name__, module, force)
            return module

        # Called as decorator: @registry.register_module(name='MyName')
        def _register(cls_or_func):
            reg_name = name if name else cls_or_func.__name__
            self._do_register(reg_name, cls_or_func, force)
            return cls_or_func

        return _register

    def _do_register(self, name: str, obj: Any, force: bool = False) -> None:
        if name in self._module_dict and not force:
            raise KeyError(
                f"'{name}' is already registered in {self._name} registry. "
                f"Use force=True to override."
            )
        self._module_dict[name] = obj

    def build(self, cfg: Dict[str, Any], **default_kwargs) -> Any:
        """Build a module from a config dict.

        Args:
            cfg: Config dict. Must contain key 'type' (str).
            **default_kwargs: Default arguments merged into cfg (cfg values take priority).

        Returns:
            Instantiated module.

        Example:
            >>> backbone = BACKBONES.build({'type': 'ResNet', 'depth': 50})
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
        if "type" not in cfg:
            raise KeyError(f"cfg must contain key 'type', but got: {cfg}")

        cfg = cfg.copy()
        obj_type = cfg.pop("type")

        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f"'{obj_type}' is not registered in {self._name} registry. "
                    f"Available: {list(self._module_dict.keys())}"
                )
        elif callable(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f"type must be a str or callable, but got {type(obj_type)}"
            )

        # default_kwargs has lower priority than cfg
        kwargs = {**default_kwargs, **cfg}
        return obj_cls(**kwargs)


# ── Global registries (one per model component type) ──────────────────────────
BACKBONES = Registry("backbone")
DECODE_HEADS = Registry("decode_head")
SEGMENTORS = Registry("segmentor")
LOSSES = Registry("loss")
