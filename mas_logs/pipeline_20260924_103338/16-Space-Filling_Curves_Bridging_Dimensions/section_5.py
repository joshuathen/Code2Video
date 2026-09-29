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
        self.setup_layout("Summary & Philosophical Takeaway", [
            "Space-filling curves are infinite limit objects.",
            "They occupy space with continuous motion.",
            "A 1D line evolving into 2D area."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Fade in infinity limit graphic [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg] #FFFFFF.
        summary_label = Text("Limit Objects", font_size=32, color=WHITE)
        self.place_at_grid(summary_label, "A5", scale_factor=0.8)
        
        # Use an SVG placeholder if asset isn't found/needed specifically
        curve_sketch = VGroup(*[Line(np.array([-0.5,0,0]), np.array([0.5,0,0])).rotate(i*30*DEGREES) for i in range(12)])
        self.place_at_grid(curve_sketch, "C3", scale_factor=0.6)
        
        self.play(FadeIn(summary_label), Create(curve_sketch))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show 1D continuous path motion [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg] #FF00FF.
        connection_arrow = Arrow(self.grid["C3"], self.grid["D3"], color="#FF00FF")
        self.add(connection_arrow)
        
        self.lecture[1].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display 1D evolving to 2D [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg] #00FFFF.
        line_obj = Line(start=LEFT, end=RIGHT, color=WHITE)
        square_obj = Square(side_length=1.5, color=WHITE)
        self.place_in_area(line_obj, "E1", "E2", scale_factor=0.6)
        self.place_in_area(square_obj, "E5", "E6", scale_factor=0.6)
        
        self.play(Transform(line_obj, square_obj))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
