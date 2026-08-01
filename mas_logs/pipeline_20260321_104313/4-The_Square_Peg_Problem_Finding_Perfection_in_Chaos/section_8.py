from manim import *
import numpy as np

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

class Section8Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        self.setup_layout(
            "Conclusion: The Power of Topology", 
            [
                "Topology finds hidden order within chaotic shapes.", 
                "Higher dimensions help us solve complex 2D puzzles.", 
                "The square peg problem remains a beautiful mystery."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(GOLD))

        # Create a "chaotic squiggle" in #C2B280
        stain_points = [
            [1.5, 0.5, 0], [1.2, 1.4, 0], [0.2, 1.8, 0], [-1.0, 1.3, 0],
            [-1.7, 0.2, 0], [-1.3, -1.0, 0], [-0.1, -1.5, 0], [1.1, -1.3, 0],
            [1.6, -0.3, 0]
        ]
        stain = VMobject(color="#C2B280")
        stain.set_points_as_corners([*stain_points, stain_points[0]])
        stain.make_smooth()
        stain.set_fill("#C2B280", opacity=0.4)
        
        # Create a "glowing" square in #FFFF00
        square = Square(side_length=1.4, color="#FFFF00").set_stroke(width=3)
        glow = square.copy().set_stroke(width=12, opacity=0.3, color="#FFFF00")
        square_group = VGroup(square, glow)
        
        # Position the stain and square in the visual area
        # Fix Issue 58: Utilize top row for better balance
        viz_container = VGroup(stain, square_group)
        self.place_in_area(viz_container, 'A2', 'D5', scale_factor=1.0)
        
        # Resolve squiggle into symmetric pattern (circle)
        self.play(Create(stain), run_time=1.0)
        
        circle_target = Circle(radius=1.3, color="#C2B280")
        circle_target.set_fill("#C2B280", opacity=0.2)
        # Fix Issue 59: Ensure circle target alignment
        self.place_in_area(circle_target, 'A2', 'D5', scale_factor=1.0)
        
        self.play(Transform(stain, circle_target), run_time=2.0)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(GOLD)
        )
        
        # Show the 2D square pulsing and glowing brightly
        self.play(FadeIn(square_group), run_time=1)
        self.play(
            square_group.animate.scale(1.15),
            rate_func=there_and_back,
            run_time=2.0
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GOLD)
        )
        
        # Final conclusion text: "Topology: Finding order within chaos"
        final_text = Text(
            "Topology: Finding order within chaos", 
            font_size=24, 
            color="#FFFFFF"
        )
        # Fix Issue 60: Move text up from F row to avoid cutoff
        self.place_in_area(final_text, 'E1', 'E6')
        
        self.play(FadeIn(final_text))
        self.wait(3)
