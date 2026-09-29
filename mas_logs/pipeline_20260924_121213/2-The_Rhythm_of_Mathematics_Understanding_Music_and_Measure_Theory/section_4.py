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
        self.setup_layout("Complex Meters and Syncopation", [
            "Compound meters use groups of three beats.",
            "Math allows us to accent specific beats.",
            "Syncopation creates rhythm by shifting musical stress."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show a 7/8 time signature meter structure using a [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg]. (Color: #FFCC00)
        self.lecture[0].set_color("#FFCC00")
        metronome = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg", color="#FFCC00")
        beats = VGroup(*[Circle(radius=0.3, color="#FFCC00") for _ in range(7)])
        beats.arrange(RIGHT, buff=0.2)
        meter_group = VGroup(metronome, beats).arrange(DOWN, buff=0.3)
        self.place_in_area(meter_group, 'B2', 'B5', scale_factor=0.7)
        self.add(meter_group)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Highlight syncopated off-beat accents. (Color: #FF6600)
        self.lecture[1].set_color("#FF6600")
        accents = VGroup(*[Star(outer_radius=0.25, inner_radius=0.1, color="#FF6600") for _ in [0, 3, 5]])
        self.place_at_grid(accents[0], 'C2', scale_factor=1.0)
        self.place_at_grid(accents[1], 'C4', scale_factor=1.0)
        self.place_at_grid(accents[2], 'C6', scale_factor=1.0)
        self.add(accents)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Visualize irregular grouping patterns in time led by a [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/conductor.svg]. (Color: #33CCFF)
        self.lecture[2].set_color("#33CCFF")
        conductor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/conductor.svg", color="#33CCFF")
        pattern = VGroup(
            Rectangle(height=0.5, width=1.0, color="#33CCFF"),
            Rectangle(height=0.5, width=1.5, color="#33CCFF"),
            Rectangle(height=0.5, width=1.0, color="#33CCFF")
        ).arrange(RIGHT, buff=0.3)
        group_vis = VGroup(conductor, pattern).arrange(DOWN, buff=0.3)
        self.place_in_area(group_vis, 'D2', 'D5', scale_factor=0.6)
        self.add(group_vis)
        self.wait(2)
