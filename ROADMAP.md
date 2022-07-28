
- [ ] 库质量维护、改善方面：Julia侧代码，清理，改善代码组织、代码注释（沟通）
- [ ] 数据转换：
  - [ ] Python侧数据传入Julia：支持更多的基本类型（例如complex)、NumPy array到Julia
的no copy转换
  - [ ] Julia array到Python的no copy转换
  - [ ] （复杂）Julia数组的view到Python的no copy转换
- [ ] 报错信息的改善


## Python使用流程的优化
    
- [ ]默认行为: 
  - [ ] 自动为Julia去安装环境管理的Julia.
  - [ ] 下载速度：自动选择最近镜像，也可以手动选择.
           
    ```python
    jnumpy.set_julia_mirror(mirror)
    ```
  - [ ] 自定义行为：用户可以自己指定使用的Julia

  - [ ] 为Julia安装TyPython依赖。

  - [ ] 使用jnumpy的用户，可以自动将当前Python包当做一个Julia project，预编译用到的Julia源代码。

    ```python
    import jnumpy as np
    np.load_c_extension(__file__) # 必须在Python主模块调用
    ```
    
