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
        lecture_lines = ["Complex waves are made of sines.", "Harmonics have different frequencies.", "Every wave is a sum."]
        self.setup_layout("Mathematical Foundation: Decomposing Signals", lecture_lines)
        
        # Define mobjects
        axes = Axes(x_range=[0, 6, 1], y_range=[-2, 2, 1], axis_config={"include_tip": False})
        complex_wave = axes.plot(lambda x: np.sin(2 * PI * x) + 0.5 * np.sin(4 * PI * x), color=WHITE)
        
        sine1 = axes.plot(lambda x: np.sin(2 * PI * x), color='#FF0000')
        sine2 = axes.plot(lambda x: 0.5 * np.sin(4 * PI * x), color='#00FF00')
        sine3 = axes.plot(lambda x: 0.25 * np.sin(6 * PI * x), color='#0000FF')
        
        sines = VGroup(sine1, sine2, sine3).arrange(DOWN, buff=0.1)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color('#FFFFFF'))
        # Using C2-F6 for visual area to avoid collision with lecture text
        self.place_in_area(axes, 'C2', 'F6', scale_factor=0.5)
        self.place_in_area(complex_wave, 'C2', 'F6', scale_factor=0.5)
        self.play(Create(axes), Create(complex_wave))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color('#FFD700'))
        self.play(FadeOut(complex_wave), ReplacementTransform(axes.copy(), VGroup(axes.copy(), sines)))
        self.play(Create(sine1), Create(sine2), Create(sine3))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color('#FFFF00'))
        self.play(FadeOut(sines), FadeIn(complex_wave))
        self.play(complex_wave.animate.set_stroke(width=4))
