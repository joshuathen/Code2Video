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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Kolmogorov Hypothesis: Self-Similarity", [
            "Kolmogorov theory explains turbulent energy cascades.",
            "Large eddies break into smaller ones.",
            "Small scales show universal statistical behavior.",
            "Defining the Kolmogorov length scale is crucial.",
            "Energy is dissipated at the smallest scales."
        ])
        
        # Assets
        eddy_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eddy.svg")
        marker_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/marker.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        cascade = VGroup(*[Circle(radius=0.3, color="#FFD700", fill_opacity=0.3) for _ in range(3)])
        self.place_in_area(cascade, "A1", "C3")
        self.add(cascade)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        eddy = self.place_at_grid(eddy_asset.copy(), "B4", scale_factor=0.5)
        eddy.set_color("#FF4500")
        self.add(eddy)
        small_eddies = VGroup(*[eddy.copy().scale(0.5) for _ in range(4)])
        self.place_in_area(small_eddies, "C4", "F6")
        self.play(ReplacementTransform(eddy, small_eddies))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00BFFF"))
        small_scales = VGroup(*[Dot(color="#00BFFF") for _ in range(10)])
        self.place_in_area(small_scales, "D1", "F3")
        self.play(FadeIn(small_scales))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#ADFF2F"))
        marker = self.place_at_grid(marker_asset.copy(), "E5", scale_factor=0.5)
        marker.set_color("#ADFF2F")
        self.play(FadeIn(marker))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        dissipation = Dot(color="#FF00FF", radius=0.2)
        self.place_at_grid(dissipation, "F6")
        self.play(Flash(dissipation))
        self.wait(2)
