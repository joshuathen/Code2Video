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
        self.setup_layout("Conclusion: The Beauty of Mathematical Equivalence", ["Pi emerges from block collisions.", "Phase space matches circular geometry.", "Math and physics are linked."])
        
        # Visualization objects
        circle = Circle(radius=1.2, color=BLUE)
        pi_label = MathTex(r"\\pi", font_size=72, color=YELLOW)
        horizontal_line = Line(start=LEFT, end=RIGHT, color=GREEN)

        # === Animation for Lecture Line 1 ===
        # Fix 1: Place circle in B4-E6 area
        self.place_in_area(circle, 'B4', 'E6', scale_factor=0.9)
        # Fix 2: Place pi_label at C5 grid
        self.place_at_grid(pi_label, 'C5', scale_factor=0.7)
        
        self.play(FadeIn(circle), Write(pi_label))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Adding a phase space arc
        arc = Arc(radius=1.2, start_angle=0, angle=PI, color=RED)
        arc.move_to(circle.get_center())
        
        self.play(Create(arc))
        self.lecture[1].set_color(RED)

        # === Animation for Lecture Line 3 ===
        # Fix 3: Place horizontal line in D3-D5 area
        self.place_in_area(horizontal_line, 'D3', 'D5', scale_factor=0.8)
        
        self.play(Create(horizontal_line))
        self.lecture[2].set_color(GREEN)
        
        self.wait(2)
