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
        self.setup_layout("The Ingredients: Linear Combinations and Span", 
                          ["Linear combinations scale and add vectors together.", 
                           "Span is the total reachable territory.", 
                           "Example: Two vectors cover the entire plane."])
        
        # Define colors for lecture lines
        colors = ["#FF7F50", "#87CEFA", "#90EE90"]
        
        # === Animation for Lecture Line 1 ===
        # Linear combinations scale and add vectors together.
        self.lecture[0].set_color(colors[0])
        v1 = Vector([1, 0.5], color=YELLOW)
        v2 = Vector([-0.5, 1], color=RED)
        self.place_at_grid(v1, "C2", scale_factor=1)
        self.place_at_grid(v2, "C3", scale_factor=1) # Fixed overlap (Issue 20, 32)
        self.play(FadeIn(v1), FadeIn(v2))
        
        # Add parallelogram asset (Issue 17)
        para = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg")
        self.place_at_grid(para, "D3", scale_factor=0.5)
        self.play(FadeIn(para))
        
        # === Animation for Lecture Line 2 ===
        # Span is the total reachable territory.
        self.lecture[1].set_color(colors[1])
        span_area = Rectangle(width=4, height=3, fill_opacity=0.3, color=BLUE, stroke_width=0)
        self.place_in_area(span_area, "C3", "E5", scale_factor=0.8) # Fixed encroachment (Issue 22, 34)
        self.play(FadeIn(span_area))
        
        # === Animation for Lecture Line 3 ===
        # Example: Two vectors cover the entire plane.
        self.lecture[2].set_color(colors[2])
        grid = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], background_line_style={"stroke_opacity": 0.2}).scale(0.5)
        self.place_at_grid(grid, "E3", scale_factor=0.6) # Fixed obstruction (Issue 21, 33)
        self.play(Create(grid))
        self.wait(2)
