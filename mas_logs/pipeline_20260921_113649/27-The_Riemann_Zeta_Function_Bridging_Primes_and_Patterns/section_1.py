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
        self.setup_layout("Prerequisites: The Power Series Foundation", 
                          ["Consider an infinite sum of powers.", 
                           "As s increases, terms shrink rapidly.", 
                           "This allows the series to converge."])
        
        # Asset Placeholder representation: Since actual SVG assets are dummy files
        # we will use generic Shapes to represent the requested assets if needed.
        # Given the instruction [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # is a "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", we treat it as an empty or neutral placeholder.

        # === Animation for Lecture Line 1 ===
        # Create a circle representing a Taylor series convergence
        circle = Circle(radius=1.0, color="#3498DB", stroke_width=4)
        # Using place_in_area as requested
        self.place_in_area(circle, 'C4', 'C5', scale_factor=0.6)
        self.play(Create(circle))
        self.lecture[0].set_color("#3498DB")

        # === Animation for Lecture Line 2 ===
        # Show an animation of terms adding up to a function
        dots = VGroup(*[Dot(color="#E74C3C") for _ in range(5)])
        dots.arrange(RIGHT, buff=0.2)
        # Using place_in_area as requested
        self.place_in_area(dots, 'E4', 'E5', scale_factor=0.8)
        
        self.play(FadeIn(dots))
        self.lecture[1].set_color("#E74C3C")

        # === Animation for Lecture Line 3 ===
        # Highlight the convergence concept
        square = Square(side_length=1.0, color="#2ECC71", fill_opacity=0.3)
        # Using place_at_grid as requested
        self.place_at_grid(square, 'C6', scale_factor=0.6)
        
        self.play(DrawBorderThenFill(square))
        self.lecture[2].set_color("#2ECC71")
        
        self.wait(2)
