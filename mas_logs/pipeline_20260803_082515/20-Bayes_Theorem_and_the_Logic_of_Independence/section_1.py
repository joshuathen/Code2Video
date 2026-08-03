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
        self.setup_layout("Introduction: The Intuition of Independence", [
            "Independence means one event doesn't affect another's probability.",
            "A cat meowing doesn't change a coin flip's outcome.",
            "We represent this visually using a probability square."
        ])
        
        # Colors
        color_a = "#ADD8E6" # Light Blue
        color_b = "#FFB6C1" # Light Pink
        color_sq = "#D3D3D3" # Light Gray
        color_inter = "#00FF00" # Green

        # === Animation for Lecture Line 1 ===
        # Step 1: Create a large unit square (#D3D3D3) representing the total probability 1.
        self.play(self.lecture[0].animate.set_color(color_sq))
        
        main_square = Square(side_length=3.0, color=color_sq, stroke_width=2)
        main_square.set_fill(color_sq, opacity=0.1)
        # Using area B2 to E5 (3x3 grid units)
        self.place_in_area(main_square, 'B2', 'E5')
        
        label_1 = Text("Total Prob = 1.0", font_size=24, color=color_sq)
        self.place_in_area(label_1, 'A3', 'A4', scale_factor=0.8) # Centered above the square
        
        self.play(Create(main_square), Write(label_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Step 2: Highlight a vertical strip 'Event A' (#ADD8E6) and a horizontal strip 'Event B' (#FFB6C1).
        self.play(self.lecture[1].animate.set_color(color_a))
        
        sq_center = main_square.get_center()
        sq_side = 3.0
        
        # Strip A (Vertical) - width 60% of square
        a_width = sq_side * 0.6
        strip_a = Rectangle(
            width=a_width, height=sq_side, 
            color=color_a, fill_color=color_a, fill_opacity=0.4, stroke_width=2
        )
        strip_a.move_to(sq_center)
        strip_a.shift(LEFT * (sq_side/2 - a_width/2))
        
        label_a = Text("Event A (Cat)", font_size=24, color=color_a)
        self.place_at_grid(label_a, 'F2', scale_factor=0.7) # Below left
        
        # Strip B (Horizontal) - height 40% of square
        b_height = sq_side * 0.4
        strip_b = Rectangle(
            width=sq_side, height=b_height, 
            color=color_b, fill_color=color_b, fill_opacity=0.4, stroke_width=2
        )
        strip_b.move_to(sq_center)
        strip_b.shift(UP * (sq_side/2 - b_height/2))
        
        label_b = Text("Event B (Coin)", font_size=24, color=color_b)
        self.place_at_grid(label_b, 'C6', scale_factor=0.7) # To the right
        
        self.play(FadeIn(strip_a), Write(label_a))
        self.play(FadeIn(strip_b), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Step 3: Highlight the intersection area (#00FF00) and label it 'Area = P(A) * P(B)' to show independence.
        self.play(self.lecture[2].animate.set_color(color_inter))
        
        # Intersection is the rectangle at top-left
        inter_rect = Rectangle(
            width=a_width, height=b_height,
            color=color_inter, fill_color=color_inter, fill_opacity=0.8, stroke_width=3
        )
        inter_rect.move_to(sq_center)
        inter_rect.shift(LEFT * (sq_side/2 - a_width/2) + UP * (sq_side/2 - b_height/2))
        
        # Independence Formula: P(A ∩ B) = P(A) * P(B)
        label_inter = MathTex("P(A \\cap B) = P(A) \\cdot P(B)", font_size=28, color=color_inter)
        self.place_in_area(label_inter, 'F3', 'F5', scale_factor=0.8) # Centered below the square
        
        self.play(Create(inter_rect), Write(label_inter))
        self.wait(2)
