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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Mathematical Bridge: The Del Operator", [
            "Del is our mathematical tool.",
            "Divergence uses the dot product.",
            "Curl uses the cross product."
        ])
        
        # Define symbols
        del_op = MathTex(r"{\nabla}", color="#FFEB3B", font_size=72)
        div_op = MathTex(r"{\nabla \cdot \mathbf{F}}", color="#4CAF50", font_size=48)
        curl_op = MathTex(r"{\nabla \times \mathbf{F}}", color="#2196F3", font_size=48)
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(del_op, "B2", scale_factor=0.8)
        self.play(Write(del_op))
        self.play(self.lecture[0].animate.set_color("#FFEB3B"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeOut(del_op))
        self.place_in_area(bridge_icon, "A2", "B3", scale_factor=0.5)
        self.play(FadeIn(bridge_icon))
        
        self.place_at_grid(div_op, "D2", scale_factor=0.8)
        self.play(Write(div_op))
        self.play(self.lecture[1].animate.set_color("#4CAF50"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(div_op), FadeOut(bridge_icon))
        self.place_at_grid(curl_op, "F2", scale_factor=0.8)
        self.play(Write(curl_op))
        self.play(self.lecture[2].animate.set_color("#2196F3"))
        self.wait(1)
        
        # Display all
        all_group = VGroup(
            MathTex(r"{\nabla}", color="#FFEB3B"),
            MathTex(r"{\nabla \cdot \mathbf{F}}", color="#4CAF50"),
            MathTex(r"{\nabla \times \mathbf{F}}", color="#2196F3")
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(all_group, "C4", "F6", scale_factor=0.9)
        self.play(FadeIn(all_group))
        self.wait(2)
