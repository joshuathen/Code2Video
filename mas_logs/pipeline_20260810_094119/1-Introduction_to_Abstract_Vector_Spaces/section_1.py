from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Leap from Concrete to Abstract", [
            "Vectors are arrows in 2D space. [Asset: arrow]", 
            "Space acts like a system of rules. [Asset: rules]", 
            "Abstract objects follow these exact rules. [Asset: objects]", 
            "We generalize beyond 2D coordinate grids. [Asset: grid]", 
            "Vector spaces define this unified structure. [Asset: structure]"
        ])
        
        # Elements
        grid_lines = NumberPlane(x_range=[-2, 2], y_range=[-2, 2]).scale(0.5)
        self.place_in_area(grid_lines, 'B2', 'E5', scale_factor=0.6)
        
        vec = Arrow(start=ORIGIN, end=RIGHT*1.5 + UP*1.0, color="#FF6600")
        vec_label = MathTex("v", color="#FF6600").next_to(vec.get_end(), UP)
        v_group = VGroup(grid_lines, vec, vec_label)
        
        abstract_symbol = Text("s", font_size=48, color="#00CCFF")
        
        # Asset for line 3
        arrow_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg")
        func_curve = VGroup(
            Axes(x_range=[0, 1], y_range=[0, 1], axis_config={"include_tip": False}).scale(0.3),
            MathTex("f(x)", color="#FFFF00")
        ).arrange(DOWN)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(v_group))
        self.lecture[0].set_color("#FF6600")

        # === Animation for Lecture Line 2 ===
        self.play(FadeOut(grid_lines), FadeOut(vec), FadeOut(vec_label), FadeIn(abstract_symbol))
        self.lecture[1].set_color("#00CCFF")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(arrow_icon, 'C3', scale_factor=0.5)
        self.play(FadeIn(arrow_icon), ReplacementTransform(abstract_symbol, func_curve))
        self.lecture[2].set_color("#FFFF00")

        # === Animation for Lecture Line 4 ===
        system_box = SurroundingRectangle(func_curve, color=WHITE, buff=0.2)
        self.play(Create(system_box))
        self.lecture[3].set_color("#FFFFFF")

        # === Animation for Lecture Line 5 ===
        self.play(Indicate(system_box))
        self.lecture[4].set_color("#FFFFFF")
