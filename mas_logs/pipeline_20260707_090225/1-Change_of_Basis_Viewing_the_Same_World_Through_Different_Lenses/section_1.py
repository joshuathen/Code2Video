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

class Section1Scene(TeachingScene):
    def construct(self):
        # Updated lecture lines per core teaching content instructions
        lecture_lines = [
            'Imagine a target located at these standard coordinates.',
            'A drone views this using a tilted internal map.',
            'One point now has two different sets of coordinates.'
        ]
        self.setup_layout("The Hook: The Misunderstood Scout Drone", lecture_lines)
        
        # Colors
        GOLD = "#FFD700"
        CYAN = "#00FFFF"
        GRAY = "#666666"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(GOLD))
        
        standard_grid = NumberPlane(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            x_length=5, y_length=5,
            background_line_style={"stroke_color": GRAY, "stroke_width": 1},
            axis_config={"stroke_color": WHITE, "stroke_width": 1.5}
        )
        # Position standard grid in the right panel using visual anchor area
        self.place_in_area(standard_grid, 'A1', 'F6')
        
        target_dot = Dot(color=GOLD, radius=0.1)
        # Position target dot visually in the upper quadrant (B4)
        self.place_at_grid(target_dot, 'B4') 
        
        self.play(Create(standard_grid))
        self.play(FadeIn(target_dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(CYAN)
        )
        
        # Tilted grid for the drone's internal perspective
        drone_grid = NumberPlane(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            x_length=5, y_length=5,
            background_line_style={"stroke_color": CYAN, "stroke_width": 1, "stroke_opacity": 0.4},
            axis_config={"stroke_color": CYAN, "stroke_width": 2}
        ).rotate(PI/6)
        self.place_in_area(drone_grid, 'A1', 'F6')
        
        self.play(FadeIn(drone_grid))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Resolved Issue 47: Updated label positions and using place_in_area to avoid crowding
        drone_label = Text("Drone: (1,2)", color=CYAN, font_size=20)
        standard_label = Text("Standard: (0,3)", color=GOLD, font_size=20)
        
        # Offsetting labels from the horizontal axis using the A-B row range (Issue 47 Fix)
        self.place_in_area(drone_label, 'A2', 'B3', scale_factor=0.8)
        self.place_in_area(standard_label, 'A5', 'B6', scale_factor=0.8)
        
        # Highlight the conflicting coordinate descriptions
        self.play(
            Write(drone_label),
            Write(standard_label)
        )
        
        # Subtle visual emphasis on the perspectival shift
        self.play(
            standard_grid.animate.set_stroke(width=3, color=WHITE),
            drone_grid.animate.set_stroke(opacity=0.2),
            run_time=1
        )
        self.play(
            standard_grid.animate.set_stroke(width=1.5, color=WHITE),
            drone_grid.animate.set_stroke(opacity=0.5),
            run_time=1
        )
        
        self.wait(2)
