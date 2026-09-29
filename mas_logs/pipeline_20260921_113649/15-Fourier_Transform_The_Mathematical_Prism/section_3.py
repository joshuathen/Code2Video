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
        lecture_lines = [
            "Time domain shows the signal.",
            "Frequency domain shows the ingredients.",
            "Scanning reveals specific frequencies.",
            "Peaks highlight dominant components.",
            "The transform links both worlds."
        ]
        self.setup_layout("The Core Concept: Domain Shifting", lecture_lines)
        
        # Animation Elements
        time_wave = FunctionGraph(lambda x: 0.3 * np.sin(4 * x) + 0.2 * np.sin(10 * x), x_range=[-2, 2], color=WHITE)
        freq_plot = Axes(x_range=[0, 15, 5], y_range=[0, 1, 0.5], axis_config={"include_tip": False}).scale(0.5)
        spike1 = Line(ORIGIN, UP * 0.5, color=YELLOW)
        spike2 = Line(ORIGIN, UP * 0.3, color=YELLOW)
        scanner = DashedVMobject(Line(UP*0.5, DOWN*0.5, color="#E74C3C"))
        legend_box = Rectangle(color=BLUE, height=0.5, width=1.0)
        
        # Placement
        self.place_in_area(time_wave, "A1", "B6", 0.8)
        self.place_in_area(freq_plot, "E1", "F6", 0.9) # Fix for 36
        self.place_in_area(scanner, "B3", "C6", 1.2)   # Fix for 35
        self.place_at_grid(legend_box, "A5", 0.7)      # Fix for 37
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(time_wave))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(Create(freq_plot))
        self.lecture[1].set_color("#3498DB")

        # === Animation for Lecture Line 3 ===
        self.play(scanner.animate.shift(RIGHT * 3))
        self.lecture[2].set_color("#E74C3C")

        # === Animation for Lecture Line 4 ===
        spike1.next_to(freq_plot.c2p(4, 0), UP, buff=0)
        spike2.next_to(freq_plot.c2p(10, 0), UP, buff=0)
        self.play(Create(spike1), Create(spike2))
        self.lecture[3].set_color(YELLOW)

        # === Animation for Lecture Line 5 ===
        arc = Arc(radius=1.0, start_angle=PI/2, angle=-PI, color=PURPLE)
        self.play(Create(arc))
        self.lecture[4].set_color(PURPLE)
        self.wait(1)
