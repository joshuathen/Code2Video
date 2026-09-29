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
            "Zeros hold the prime secret.",
            "They live in the strip.",
            "The critical line is key.",
            "All non-trivial zeros lie here.",
            "This defines prime number distribution."
        ]
        self.setup_layout("The Riemann Hypothesis: The Critical Strip", lecture_lines)
        
        # Assets (Using placeholders as per instructions: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg)
        # Assuming we need to load them but can't display actual images, we'll create placeholders
        def create_asset_placeholder(text):
            return Text(text, font_size=12, color=GRAY)
        
        icon1 = create_asset_placeholder("[Icon]")
        self.place_at_grid(icon1, 'A1')
        
        # Visualization objects
        strip = Rectangle(height=4, width=0.8, color="#E67E22", fill_opacity=0.3)
        self.place_in_area(strip, 'B3', 'E4', scale_factor=0.9)
        
        line = Line(start=UP*2, end=DOWN*2, color="#FFFFFF")
        self.place_at_grid(line, 'C3', scale_factor=0.85)
        
        zeros = VGroup(*[Dot(color="#D35400", radius=0.1) for _ in range(5)])
        self.place_at_grid(zeros, 'B3', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#E67E22"))
        self.play(Create(strip))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(Create(line))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#D35400"))
        self.play(FadeIn(zeros))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#F1C40F"))
        self.wait(2)
