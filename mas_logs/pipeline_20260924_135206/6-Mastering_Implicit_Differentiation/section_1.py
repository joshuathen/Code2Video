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
        self.setup_layout("The 'Why': Explicit vs. Implicit Curves", [
            "Explicit functions like y=f(x) are easy.", 
            "Implicit curves hide variables like y=y(x).", 
            "Sometimes solving for y is impossible."
        ])
        
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True}).scale(0.6)
        
        # === Animation for Lecture Line 1 ===
        explicit_curve = axes.plot(lambda x: 0.5 * x, color="#FFFFFF")
        func_label = MathTex(r"y = f(x)", color="#FFFFFF")
        
        self.play(Create(axes), Write(explicit_curve))
        # Fixed per issue 25/40
        self.place_at_grid(func_label, "D3", scale_factor=0.7)
        self.play(Write(func_label))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        implicit_curve = Circle(radius=1.2, color="#FF0000").move_to(axes.c2p(0, 0))
        implicit_label = MathTex(r"x^2 + y^2 = r^2", color="#FF0000")
        
        self.play(FadeOut(explicit_curve), FadeOut(func_label))
        self.play(Create(implicit_curve))
        # Fixed per issue 26/41
        self.place_at_grid(implicit_label, "D5", scale_factor=0.7)
        self.play(Write(implicit_label))
        self.lecture[1].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
