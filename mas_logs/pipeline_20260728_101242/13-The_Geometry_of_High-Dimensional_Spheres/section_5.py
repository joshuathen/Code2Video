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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        self.setup_layout("Applications: The Curse of Dimensionality", [
            "In high dimensions, points are usually far apart.",
            "This makes searching and organizing big data difficult.",
            "Geometry explains the challenges of modern AI models."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Create a cluster of 20 white dots (#FFFFFF) at the center of the screen, 
        # incorporating the data icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/data.svg]
        self.lecture[0].set_color(WHITE) # Match white dots
        
        data_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/data.svg")
        self.place_in_area(data_icon, "C3", "D4", scale_factor=0.6)
        
        num_points = 20
        points = VGroup(*[Dot(radius=0.06, color="#FFFFFF") for _ in range(num_points)])
        
        # Define the center for points (same as data_icon)
        center = data_icon.get_center()
        
        # Generate random initial offsets within a small cluster
        np.random.seed(42)
        initial_offsets = [
            np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0])
            for _ in range(num_points)
        ]
        
        # Initial placement of points
        for i, p in enumerate(points):
            p.move_to(center + initial_offsets[i])
            
        self.play(FadeIn(data_icon), FadeIn(points))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the dots moving rapidly toward the edges of the frame to represent increased distance in high-D.
        # Highlight current line
        self.lecture[1].set_color(YELLOW)
        
        # ValueTracker to control the "expansion"
        spread_tracker = ValueTracker(1.0)
        
        # Add updaters for persistent movement
        def update_point_pos(p, idx):
            p.move_to(center + initial_offsets[idx] * spread_tracker.get_value())

        for i, p in enumerate(points):
            p.add_updater(lambda m, i=i: update_point_pos(m, i))
            
        # Move dots to edges and fade out data icon
        self.play(
            spread_tracker.animate.set_value(5.0),
            FadeOut(data_icon),
            run_time=3,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fade in the text 'Curse of Dimensionality' in bold red (#FF0000) 
        # along with the robot icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg].
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF0000") # Match red title
        
        curse_title = Text("Curse of Dimensionality", weight=BOLD, color="#FF0000", font_size=32)
        # Fix for Issue 32: place in area F2-F5
        self.place_in_area(curse_title, 'F2', 'F5', scale_factor=0.7)
        
        robot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        # Place robot icon above the text to avoid overlap with spread points
        self.place_in_area(robot_icon, "C3", "D4", scale_factor=0.8)
        
        self.play(FadeIn(curse_title), FadeIn(robot_icon))
        self.wait(3)

        # Final color reset and hold
        self.lecture[2].set_color(WHITE)
        # Cleanup updaters
        for p in points:
            p.remove_updater(update_point_pos)
        self.wait(2)
