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
        lecture_lines = [
            "Music relies on a steady, consistent beat.",
            "This pulse acts like a clock for musicians.",
            "Think of a heartbeat keeping rhythm steady."
        ]
        self.setup_layout("The Prerequisite: The Metronome Pulse", lecture_lines)
        
        # Assets
        metronome_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg")
        heart_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/heart.svg")
        
        # Define elements
        bars = VGroup(*[Line(UP, DOWN, color=WHITE) for _ in range(5)]).arrange(RIGHT, buff=0.5)
        self.place_at_grid(metronome_icon, 'B4', scale_factor=0.6)
        
        timeline = Line(LEFT*2.5, RIGHT*2.5, color=WHITE)
        self.place_at_grid(timeline, 'D4', scale_factor=0.7)
        
        pulse_dot = heart_icon
        pulse_dot.set_color("#00FFFF")
        pulse_dot.move_to(timeline.get_left())
        
        border = SurroundingRectangle(timeline, color="#FFD700", buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.add(metronome_icon)
        self.play(Create(bars))
        self.play(self.lecture[0].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 2 ===
        for bar in bars:
            self.play(bar.animate.set_color("#FF00FF"), run_time=0.2)
            self.play(bar.animate.set_color(WHITE), run_time=0.2)
        
        self.play(FadeOut(metronome_icon), Transform(bars, timeline))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 3 ===
        self.add(pulse_dot)
        self.play(MoveAlongPath(pulse_dot, timeline), run_time=2, rate_func=linear)
        self.play(Create(border))
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.wait(1)
