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
        self.setup_layout("Application: The Detective Case", ["Detectives use Bayes to solve cases.", "Evidence helps refine suspect probability.", "Dependent evidence provides crucial updates."])
        self.lecture.set_opacity(0)
        
        # --- Animation for Lecture Line 1 ---
        self.play(FadeIn(self.lecture[0]))
        self.lecture[0].set_color("#44AAFF")
        detective = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/detective.svg")
        detective.set_color("#44AAFF")
        self.place_at_grid(detective, 'B5', scale_factor=1.2)
        self.play(FadeIn(detective))
        
        # --- Animation for Lecture Line 2 ---
        self.play(FadeIn(self.lecture[1]))
        self.lecture[1].set_color("#FFCC00")
        footprints = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/footprints.svg")
        footprints.set_color("#FFCC00")
        self.place_at_grid(footprints, 'D2', scale_factor=1.0)
        self.play(FadeIn(footprints))
        
        # --- Animation for Lecture Line 3 ---
        self.play(FadeIn(self.lecture[2]))
        self.lecture[2].set_color("#FF5555")
        bar = Rectangle(height=0.3, width=0, color="#FF5555", fill_opacity=1)
        self.place_in_area(bar, 'E4', 'F6', scale_factor=0.8)
        self.play(Create(bar))
        self.play(bar.animate.set_width(2.0))
