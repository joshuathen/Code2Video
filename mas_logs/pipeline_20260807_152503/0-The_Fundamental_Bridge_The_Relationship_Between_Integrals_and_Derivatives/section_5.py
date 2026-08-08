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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Compute areas without infinite sums.", "Use anti-derivatives for simpler results.", "Subtraction solves the definite integral."]
        self.setup_layout("Summary & Application", lecture_lines)
        
        # Prepare animation elements
        # 1. Summary text
        summary = Text("Fundamental Theorem of Calculus", font_size=24, color=WHITE)
        self.place_at_grid(summary, "A3", scale_factor=1.0)
        
        # 2. Calculation visual (simplified)
        calc_vgroup = VGroup(
            MathTex(r"\int_{a}^{b} f(x) dx = F(b) - F(a)"),
            Text("Anti-derivative", font_size=18, color="#00FFCC")
        ).arrange(DOWN)
        self.place_in_area(calc_vgroup, "C3", "D6", scale_factor=0.9)

        # 3. Bridge icon (Using SVGMobject for asset)
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        bridge_label = Text("Bridge: Integral <-> Derivative", font_size=20, color="#FF00FF")
        bridge_group = VGroup(bridge_icon, bridge_label).arrange(DOWN)
        self.place_at_grid(bridge_group, "E4", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(FadeIn(summary))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFCC"))
        self.play(FadeIn(calc_vgroup))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.play(FadeIn(bridge_group))
        self.wait(2)
