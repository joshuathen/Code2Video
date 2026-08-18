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
        lecture_lines = ["We organize beats into measures.", "Time signatures define these boxes.", "Four beats fit in one measure."]
        self.setup_layout("Introduction to Measure (Bars)", lecture_lines)
        
        # Load assets
        metronome_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg")
        sheet_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sheet.svg")

        # Create elements
        bar_line1 = Line(UP, DOWN, color=WHITE).set_length(3)
        bar_line2 = Line(UP, DOWN, color=WHITE).set_length(3)
        beats = VGroup(*[Circle(radius=0.3, color="#00FF7F", fill_opacity=0.5) for _ in range(4)])
        beat_label = Text("Beat", font_size=20, color=WHITE)
        
        # Container for measure
        measure_box = VGroup(bar_line1, bar_line2, beats).arrange(RIGHT, buff=0.5)
        self.place_in_area(measure_box, 'B3', 'D5', scale_factor=0.6)
        
        # Setup specific assets/labels
        vertical_bar = VGroup(bar_line1, metronome_icon)
        self.place_at_grid(vertical_bar, 'C4', scale_factor=0.7)
        self.place_at_grid(beat_label, 'C5', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(bar_line1), FadeIn(metronome_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        for beat in beats:
            self.play(FadeIn(beat, scale=0.5), run_time=0.3)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(Create(bar_line2), FadeIn(sheet_icon))
        self.wait(1)
