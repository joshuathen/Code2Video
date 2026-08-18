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
        lecture_lines = [
            "Vectors are directed arrows in 2D space.",
            "Vector addition uses the tip-to-tail method.",
            "Scalar multiplication scales the arrow's length."
        ]
        self.setup_layout("Prerequisite Review: Vectors as Arrows", lecture_lines)
        
        # 1. Improved axes placement
        axes = Axes(x_range=[-1, 5], y_range=[-1, 5], axis_config={"include_tip": True})
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.5)
        self.add(axes)

        # 2. Setup Animation Elements (Using placeholder SVGs as assets if applicable)
        # Note: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg doesn't exist, 
        # but the prompt demands using it as an asset reference.
        # Since standard Manim doesn't treat .svg this way, we'll create the objects required.
        v = Arrow(start=ORIGIN, end=axes.c2p(2, 3), color=WHITE, buff=0)
        v_label = MathTex("v", color=WHITE)
        
        # 3. Label and Component Setup
        line_x = DashedLine(axes.c2p(2, 3), axes.c2p(2, 0), color=GREEN)
        line_y = DashedLine(axes.c2p(2, 3), axes.c2p(0, 3), color=GREEN)
        x_label = MathTex("x", color=GREEN)
        y_label = MathTex("y", color=GREEN)
        
        animation_group = VGroup(line_x, line_y, x_label, y_label)
        self.place_in_area(animation_group, 'B2', 'E3', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(v), Write(v_label))
        # Place label explicitly
        self.place_at_grid(v_label, 'D4', scale_factor=0.7)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Create(line_x), Create(line_y), Write(x_label), Write(y_label))
        # Place labels explicitly
        self.place_at_grid(y_label, 'B3', scale_factor=0.6)
        self.place_at_grid(x_label, 'E4', scale_factor=0.6)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        v_scaled = Arrow(start=ORIGIN, end=axes.c2p(1, 1.5), color=BLUE, buff=0)
        self.play(Transform(v, v_scaled))
        self.play(FadeOut(line_x), FadeOut(line_y), FadeOut(x_label), FadeOut(y_label))
        
        self.wait(2)
