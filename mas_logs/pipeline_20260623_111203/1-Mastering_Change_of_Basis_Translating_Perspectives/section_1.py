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
        # Setup layout with title and lecture lines
        self.setup_layout(
            "The Hook: The Owl and the Drone",
            [
                "Meet the Owl, watching from a steady, upright branch.",
                "Meet the Drone, flying at a tilted forty-five degrees.",
                "They both spot the same mouse on the ground.",
                "The Owl sees coordinates (1, 1) on his grid.",
                "To the Drone, that same mouse is at (1.41, 0)."
            ]
        )
        
        # Define shared colors
        owl_color = "#A52A2A"
        drone_color = "#00FFFF"
        mouse_color = "#FFD700"
        grid_color = "#444444"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(owl_color)
        
        # Create Owl's standard grid
        owl_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_color": grid_color, "stroke_opacity": 0.8},
            axis_config={"stroke_color": grid_color, "stroke_width": 2}
        )
        # Issue 32: Apply scale_factor=0.8 to reduce density
        self.place_in_area(owl_grid, "A1", "F6", scale_factor=0.8)
        
        # Owl icon (Triangle at origin - center of C3-D4)
        owl_icon = Triangle(color=owl_color, fill_opacity=1)
        self.place_in_area(owl_icon, 'C3', 'D4', scale_factor=0.15)
        
        self.play(FadeIn(owl_grid), FadeIn(owl_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(drone_color)
        
        # Create Drone's tilted grid
        drone_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_color": drone_color, "stroke_opacity": 0.3},
            axis_config={"stroke_color": drone_color, "stroke_opacity": 0.5, "stroke_width": 2}
        )
        drone_grid.rotate(45 * DEGREES)
        # Issue 32: Apply scale_factor=0.8 to reduce density
        self.place_in_area(drone_grid, "A1", "F6", scale_factor=0.8)
        
        # Drone icon (Diamond at origin - center of C3-D4)
        drone_icon = Square(color=drone_color, fill_opacity=0.8).rotate(45*DEGREES)
        self.place_in_area(drone_icon, 'C3', 'D4', scale_factor=0.12)
        
        self.play(Create(drone_grid), FadeIn(drone_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(mouse_color)
        
        # Mouse icon at (1,1) relative to Owl's coordinate system center
        # Intersection of B4, B5, C4, C5 corresponds to (1,1) in the visual grid
        mouse = Dot(color=mouse_color)
        self.place_in_area(mouse, 'B4', 'C5', scale_factor=1.25) # Scale factor to match radius 0.1 approx
        
        self.play(FadeIn(mouse))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(owl_color)
        
        # Owl perspective label
        owl_label = Text("(1, 1)", color=owl_color, font_size=24)
        # Issue 30: Use place_at_grid 'B5' to avoid overlap
        self.place_at_grid(owl_label, 'B5', scale_factor=0.6)
        
        self.play(Write(owl_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(drone_color)
        
        # Drone perspective label
        drone_label = Text("(1.41, 0)", color=drone_color, font_size=24)
        # Issue 31: Use place_at_grid 'C5' to avoid overlap
        self.place_at_grid(drone_label, 'C5', scale_factor=0.6)
        
        self.play(Write(drone_label))
        self.wait(2)
