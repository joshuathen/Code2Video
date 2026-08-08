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

class Section1Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "The Goal: The Error Gap"
        lines = [
            "Neural networks learn by adjusting thousands of tiny knobs.",
            "These knobs are weights that control the network's output.",
            "Our goal is to minimize the Error Gap, or Loss."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_WEIGHTS = "#00BFFF"
        COLOR_ERROR = "#FF0000"
        
        # === Animation for Lecture Line 1 ===
        # Show a stylized machine with several [Asset: knob.svg] labeled Weights (#00BFFF)
        self.lecture[0].set_color(COLOR_WEIGHTS)
        
        # Issue 26: machine_rect at B3 to E5, scale 0.9
        machine_rect = RoundedRectangle(corner_radius=0.2, height=3, width=2.5, color=GRAY, fill_opacity=0.2)
        self.place_in_area(machine_rect, "B3", "E5", scale_factor=0.9)
        
        # Issue 20: Use the knob.svg asset
        knobs = VGroup()
        for _ in range(6):
            knob = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg")
            knob.set_color(COLOR_WEIGHTS)
            knobs.add(knob)
        
        knobs.arrange_in_grid(rows=2, cols=3, buff=0.5)
        self.place_in_area(knobs, "B3", "E5", scale_factor=0.6)
        
        # Issue 24: weights_label at F4, scale 0.7
        weights_label = Text("Weights", font_size=20, color=COLOR_WEIGHTS)
        self.place_at_grid(weights_label, "F4", scale_factor=0.7)
        
        self.play(Create(machine_rect), FadeIn(knobs), Write(weights_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # These knobs are weights that control the network's output
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_WEIGHTS)
        
        # Animate rotation to show "control" before clearing space
        self.play(*[knob.animate.rotate(PI/2) for knob in knobs], run_time=1)
        
        # Issue 24: Fade out machine, knobs, and label to clear space for bullseye
        self.play(
            FadeOut(machine_rect), 
            FadeOut(knobs), 
            FadeOut(weights_label),
            run_time=0.8
        )
        
        # Issue 25: bullseye at E4, scale 0.6
        bullseye = VGroup(
            Circle(radius=0.6, color=RED, fill_opacity=1),
            Circle(radius=0.4, color=WHITE, fill_opacity=1),
            Circle(radius=0.2, color=RED, fill_opacity=1)
        )
        self.place_at_grid(bullseye, "E4", scale_factor=0.6)
        
        # Arrow hitting 2m above it (hitting at C4 while target is at E4)
        arrow_start = self.grid["A4"]
        arrow_end = self.grid["C4"]
        arrow = Arrow(start=arrow_start, end=arrow_end, color=WHITE, buff=0)
        
        self.play(Create(bullseye))
        self.play(GrowArrow(arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Our goal is to minimize the Error Gap, or Loss
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_ERROR)
        
        # Red bracket between arrow hit (C4) and bullseye center (E4)
        error_brace = BraceBetweenPoints(self.grid["C4"], self.grid["E4"], color=COLOR_ERROR, direction=RIGHT)
        error_label = Text("Error Gap", font_size=20, color=COLOR_ERROR)
        error_label.next_to(error_brace, RIGHT, buff=0.1)
        
        self.play(Create(error_brace), Write(error_label))
        self.wait(2)
