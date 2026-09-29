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
        lecture_lines = [
            "Divergence shows expansion or compression.",
            "Curl shows swirling or rotation.",
            "Both characterize vector field behavior."
        ]
        self.setup_layout("Visual Synthesis: Divergence vs. Curl", lecture_lines)
        
        # Load assets
        faucet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/faucet.svg")
        turbine = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/turbine.svg")
        
        divergence_label = Text("Divergence", font_size=24, color=RED)
        curl_label = Text("Curl", font_size=24, color=GREEN)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        self.place_at_grid(faucet, "D2", scale_factor=0.6)
        self.place_at_grid(divergence_label, "C2", scale_factor=0.7)
        self.play(FadeIn(faucet), Write(divergence_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.place_at_grid(turbine, "D5", scale_factor=0.6)
        self.place_at_grid(curl_label, "C5", scale_factor=0.7)
        self.play(FadeIn(turbine), Write(curl_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(1)
