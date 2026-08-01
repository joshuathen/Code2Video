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
        self.setup_layout(
            "The Cartesian Language: Plotting the Arrow",
            [
                "We plot vectors on a 2D coordinate plane.",
                "The tail starts at the origin zero, zero.",
                "The head points to specific x and y coordinates."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Display a 2D coordinate grid with labeled origin (0,0).
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create a plane.
        # Adjusted x_range and y_range to be more appropriate for a [3,4] vector.
        plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=4.5,
            y_length=4.5,
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_numbers": True, "font_size": 18}
        )
        # Fix: Using 'F6' instead of 'E6' to avoid vertical cramping (Issue 26)
        self.place_in_area(plane, "B2", "F6")
        
        origin_label = MathTex("(0,0)", font_size=20, color=WHITE)
        # Position origin label near origin (0,0) of the plane
        origin_label.next_to(plane.c2p(0, 0), DL, buff=0.1)
        
        self.play(Create(plane), Write(origin_label), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The tail starts at the origin zero, zero.
        # Animate a drone asset moving 3 units right and 4 units up.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Drone asset (Issue 20)
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg")
        drone.set_color(BLUE)
        drone.scale(0.3)
        
        # Start at origin
        drone.move_to(plane.c2p(0, 0))
        self.play(FadeIn(drone))
        
        # Move 3 units right
        self.play(drone.animate.move_to(plane.c2p(3, 0)), run_time=1)
        # Move 4 units up
        self.play(drone.animate.move_to(plane.c2p(3, 4)), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The head points to specific x and y coordinates.
        # Draw a green arrow (#00FF00) from (0,0) to (3,4) labeled '[3, 4]'.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        vector_arrow = Arrow(
            start=plane.c2p(0, 0),
            end=plane.c2p(3, 4),
            buff=0,
            color="#00FF00",
            stroke_width=6
        )
        
        vector_label = MathTex("[3, 4]", font_size=24, color="#00FF00")
        # Position label near the head of the vector
        vector_label.next_to(plane.c2p(3, 4), UR, buff=0.1)
        
        self.play(
            GrowArrow(vector_arrow),
            Write(vector_label),
            run_time=1.5
        )
        self.wait(2)
        
        # Return last line to white
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
