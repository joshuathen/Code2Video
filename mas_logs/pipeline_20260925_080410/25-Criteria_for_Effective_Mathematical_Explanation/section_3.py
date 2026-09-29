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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Criterion 2: The Power of Multiple Representations", [
            "Use the Rule of Three strategy.", 
            "Represent concepts algebraically, geometrically, and contextually.", 
            "Multiple representations deepen true understanding."
        ])
        
        # Elements
        alg_eq = MathTex("f(x) = 2x", color="#FF9900")
        axes = Axes(x_length=2, y_length=2, x_range=[0, 2], y_range=[0, 2]).set_color("#FF9900")
        line = axes.plot(lambda x: 2*x, color="#FF9900")
        geo_group = VGroup(axes, line)
        ctx_text = Text("Robot moves 2x speed", color="#FF9900", font_size=20)
        
        # SVGs as placeholders
        icon_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # Layout initial placement
        self.place_at_grid(alg_eq, 'B2', scale_factor=1.2)
        self.place_at_grid(geo_group, 'B4', scale_factor=0.8)
        self.place_at_grid(ctx_text, 'B6', scale_factor=0.8)
        self.place_at_grid(icon_svg, 'A5', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(alg_eq), FadeIn(geo_group), FadeIn(ctx_text), FadeIn(icon_svg))
        self.play(self.lecture[0].animate.set_color("#FF9900"))

        # === Animation for Lecture Line 2 ===
        # Fixes for obstruction as per criticism:
        self.place_in_area(alg_eq, 'C4', 'D6', scale_factor=0.8)
        self.place_in_area(geo_group, 'E1', 'F3', scale_factor=0.9)
        
        self.play(
            Rotate(alg_eq, angle=PI/4),
            Rotate(geo_group, angle=-PI/4),
            Rotate(ctx_text, angle=PI/2),
            self.lecture[1].animate.set_color("#99FF00")
        )

        # === Animation for Lecture Line 3 ===
        consolidated = VGroup(alg_eq.copy(), geo_group.copy(), ctx_text.copy(), icon_svg.copy()).arrange(DOWN).scale(0.5)
        # Fix for consolidation positioning as per criticism:
        self.place_at_grid(consolidated, 'D4', scale_factor=0.7)
        self.play(
            ReplacementTransform(VGroup(alg_eq, geo_group, ctx_text, icon_svg), consolidated),
            self.lecture[2].animate.set_color("#0099FF")
        )
        self.wait(2)
