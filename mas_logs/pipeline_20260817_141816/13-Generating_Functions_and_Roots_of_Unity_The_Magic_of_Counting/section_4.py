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
        lecture_lines = [
            "Complex roots act as a sieve.",
            "Rotating vectors isolate valid combinatorial states.",
            "This technique simplifies complex counting problems."
        ]
        self.setup_layout("Synthesis and Geometric Intuition", lecture_lines)
        
        # Load asset
        filter_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg")
        
        # Animation setup
        circle = Circle(radius=1.5, color=WHITE)
        self.place_in_area(circle, 'A3', 'F5', scale_factor=0.9)
        
        vectors = VGroup()
        for i in range(8):
            angle = i * PI / 4
            vec = Line(start=ORIGIN, end=1.5 * np.array([np.cos(angle), np.sin(angle), 0]), color=BLUE)
            vectors.add(vec)
        self.place_in_area(vectors, 'A3', 'F5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        filter_overlay = filter_icon.copy()
        self.place_in_area(filter_overlay, 'B3', 'E4', scale_factor=0.3)
        filter_overlay.set_color("#FFFFFF")
        self.play(Create(circle), Create(vectors), FadeIn(filter_overlay))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        # Flash effect on vectors
        self.play(Rotate(vectors, angle=PI/4, run_time=1.5), vectors.animate.set_color("#FF0000"), run_time=1.5)
        self.play(vectors.animate.set_color(BLUE), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        final_vec = Line(start=ORIGIN, end=np.array([1.5, 0, 0]), color="#FFFF00", stroke_width=8)
        self.place_at_grid(final_vec, 'C4', scale_factor=0.5)
        
        filter_highlight = filter_icon.copy()
        self.place_in_area(filter_highlight, 'B3', 'E4', scale_factor=0.3)
        filter_highlight.set_color("#FFFF00")
        
        self.play(FadeOut(vectors), FadeOut(circle), FadeOut(filter_overlay), FadeIn(final_vec), FadeIn(filter_highlight))
