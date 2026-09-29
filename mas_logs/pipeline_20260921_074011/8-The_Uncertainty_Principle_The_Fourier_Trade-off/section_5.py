from manim import *
import os

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
        self.setup_layout("Real-world Application: Modern Communication", [
            "We cannot localize perfectly in both.",
            "Resolution choice depends on specific goals.",
            "Radar systems exemplify this critical trade-off.",
            "Bats use pulses for different tasks.",
            "Choosing resolution governs modern signal processing."
        ])
        
        # Assets
        radar_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/radar.svg"
        
        # Elements
        if os.path.exists(radar_path):
            radar_icon = SVGMobject(radar_path, color="#33A1FF")
        else:
            radar_icon = Circle(color="#33A1FF", radius=0.5)

        spectrum = Axes(x_range=[0, 10, 1], y_range=[0, 2, 0.5], axis_config={"include_numbers": False}).scale(0.5)
        graph = spectrum.plot(lambda x: np.sin(x) * np.exp(-0.2*x) + 1, color=WHITE)
        atoms = VGroup(*[Circle(radius=0.15, color="#A133FF").move_to(spectrum.c2p(i, 1.2)) for i in range(1, 9, 2)])
        reconstructed = VGroup(*[Dot(point=spectrum.c2p(i, 1.2), color="#33FF33") for i in range(1, 9, 2)])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#33A1FF")
        self.place_in_area(spectrum, 'B3', 'D6', scale_factor=0.65)
        self.place_at_grid(radar_icon, "A3", scale_factor=0.5)
        self.play(Create(spectrum), Create(graph), FadeIn(radar_icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33A1FF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#33A1FF")
        self.play(FadeIn(atoms))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#33A1FF")
        self.play(atoms.animate.set_color("#A133FF"))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#33FF33")
        self.play(ReplacementTransform(atoms.copy(), reconstructed))
        self.wait(2)
