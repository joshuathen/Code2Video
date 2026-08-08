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
        self.setup_layout("Energy Spectrum in Turbulence", [
            "Plot energy density versus wavenumber on log-log.",
            "The inertial range shows a minus 5/3 slope.",
            "Three regimes: energy containing, inertial, and dissipation.",
            "Slope steepens in the viscous dissipation range.",
            "Spectra reveal governed processes, not pure randomness."
        ])
        
        # Create axes for energy spectrum
        axes = Axes(
            x_range=[0, 5, 1], y_range=[0, 5, 1],
            x_length=4, y_length=4,
            axis_config={"include_numbers": False, "color": WHITE}
        )
        # Applying critique: self.place_in_area(axes, 'B2', 'E5', scale_factor=0.85)
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.85)
        self.add(axes)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Create(axes))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        # 5/3 slope line
        slope_line = Line(start=axes.c2p(0.5, 4), end=axes.c2p(3, 0.5), color=GREEN)
        self.play(Create(slope_line))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Highlight regions
        reg1 = SurroundingRectangle(axes, color=YELLOW, buff=0.1)
        self.play(Create(reg1))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED)
        # Steepened slope
        steep_line = Line(start=axes.c2p(3, 0.5), end=axes.c2p(4.5, 0.1), color=RED)
        self.play(Create(steep_line))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        # Applying critique: self.place_at_grid(note, 'E4', scale_factor=0.9)
        note = Text("Governed Processes", font_size=18, color=PURPLE)
        self.place_at_grid(note, 'E4', scale_factor=0.9)
        self.play(Write(note))
        self.wait(2)
