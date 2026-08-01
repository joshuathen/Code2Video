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
        title = "Prerequisite: The Probability Square"
        lines = [
            "Imagine all possibilities fitting inside a unit square.",
            "The square's total area represents absolute certainty.",
            "Your initial belief is a portion of this space."
        ]
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # Draw a white 1x1 square representing all possibilities (#FFFFFF).
        # Grid range B2 to E5 gives a nice square area (width 3, height 3).
        # Center: (3.0, -0.3)
        unit_square = Square(side_length=3.0, color="#FFFFFF", stroke_width=2)
        self.place_in_area(unit_square, 'B2', 'E5')
        
        self.play(Create(unit_square))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        
        # Pulse the square to emphasize total area.
        self.play(Indicate(unit_square, color="#FFFFFF")) # [L004]
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3 - Blue to match the "Prior" column
        self.play(self.lecture[2].animate.set_color("#0000FF"))
        
        # Shade a blue vertical column for 20% 'Prior' probability (#0000FF).
        # Square width is 3.0, 20% is 0.6.
        prior_width = 3.0 * 0.2
        prior_rect = Rectangle(
            width=prior_width, 
            height=3.0, 
            fill_color="#0000FF", 
            fill_opacity=0.5, 
            stroke_width=0
        )
        
        # Position the rect on the left side of the square.
        # Square center is at (3.0, -0.3), left edge is 1.5.
        # Rect center should be at 1.5 + (0.6/2) = 1.8.
        prior_rect.move_to(unit_square.get_left() + RIGHT * (prior_width / 2))
        
        # Add a label 'P(Rain) = 0.2' and the rain icon (#FFFFFF).
        # [L022] Simple MathTex
        prior_label = MathTex("P(\\text{Rain}) = 0.2", color="#FFFFFF", font_size=24)
        # Fix overlap per Issue 30
        self.place_in_area(prior_label, 'C1', 'D1', scale_factor=0.6)
        
        # Rain icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/rain.svg]
        rain_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rain.svg")
        rain_icon.set_fill(color="#FFFFFF", opacity=1.0) # Ensure it is white
        # Position inside the blue column (x in [1.5, 2.1])
        # Grid area C2 to D3 center is (2.0, -0.3), which is inside.
        self.place_in_area(rain_icon, 'C2', 'D3', scale_factor=0.3)
        
        self.play(FadeIn(prior_rect))
        self.play(FadeIn(rain_icon), Write(prior_label))
        self.wait(2.0)
