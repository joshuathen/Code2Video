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
        self.setup_layout("Application: The Saccharimeter", [
            "Chemists use this to measure sugar concentrations.",
            "The saccharimeter works like a light-based scale.",
            "It reveals liquid purity through optical rotation."
        ])

        # === Animation for Lecture Line 1 ===
        # Using SVG asset
        sample_vessel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vessel.svg", color="#795548")
        sample_label = Text("Sample", font_size=20, color="#795548")
        sample_group = VGroup(sample_vessel, sample_label).arrange(DOWN)
        self.place_in_area(sample_group, 'B2', 'D2', scale_factor=0.6)
        self.lecture[0].set_color("#795548")
        self.play(FadeIn(sample_group))

        # === Animation for Lecture Line 2 ===
        # Using SVG asset
        light_ray = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg", color="#FFEB3B")
        rotation_label = Text("Rotation", font_size=20, color="#FFEB3B")
        self.place_at_grid(light_ray, 'D5', scale_factor=0.8)
        self.place_at_grid(rotation_label, 'C5', scale_factor=0.8)
        self.lecture[1].set_color("#FFEB3B")
        self.play(Create(light_ray), Write(rotation_label))

        # === Animation for Lecture Line 3 ===
        # Using SVG asset
        scale_display = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg", color="#FFFFFF")
        conc_label = Text("Concentration", font_size=20, color="#FFFFFF")
        scale_group = VGroup(scale_display, conc_label).arrange(DOWN)
        self.place_at_grid(scale_group, 'E3', scale_factor=0.7)
        self.lecture[2].set_color("#FFFFFF")
        self.play(FadeIn(scale_group))
        self.wait(2)
