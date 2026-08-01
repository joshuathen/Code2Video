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

class Section4Scene(TeachingScene):
    def construct(self):
        title_str = "Application: The Autonomous Drone Path"
        lecture_lines = [
            "The equation dx/dt = Ax defines the system's change.",
            "The solution uses the matrix exponential e^(At).",
            "Aero-Bot begins at its initial position x(0).",
            "The drone follows a path determined by the exponential.",
            "This visualizes the continuous evolution of the system's state."
        ]
        self.setup_layout(title_str, lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        eq1 = Text("dx/dt = Ax", color="#FFFFFF", font_size=32)
        # Fix Issue 32: Use B3-B5 for better centering
        self.place_in_area(eq1, "B3", "B5", scale_factor=0.9)
        self.play(Write(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        eq2 = Text("x(t) = e^(At) x(0)", color="#FFFF00", font_size=32)
        # Fix Issue 32: Use B3-B5 for better centering
        self.place_in_area(eq2, "B3", "B5", scale_factor=0.9)
        self.play(ReplacementTransform(eq1, eq2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        
        # Setup Axes on the right side grid area
        axes = Axes(
            x_range=[0, 5, 1], y_range=[0, 5, 1],
            x_length=3, y_length=3,
            axis_config={"color": WHITE},
            tips=False
        )
        # Fix Issue 33: Use E2-F6 for better layout
        self.place_in_area(axes, "E2", "F6", scale_factor=0.8)
        
        # Fix Issue 24: Use drone SVG asset
        drone_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/drone.svg"
        drone = SVGMobject(drone_path)
        drone.set_color(WHITE)
        drone.scale(0.3) # Scale relative to grid size
        
        # Position drone at origin of the axes
        drone.move_to(axes.c2p(0, 0))
        
        self.play(Create(axes), FadeIn(drone))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FFFF")
        
        # Exponential growth curve: y = a(1 - e^-bx)
        # We use a path that starts at origin
        path = axes.plot(lambda x: 4 * (1 - np.exp(-0.8*x)), x_range=[0, 4.5], color="#00FFFF")
        
        self.play(
            Create(path),
            MoveAlongPath(drone, path),
            run_time=4,
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FFFF")
        
        path_label = Text("State Evolution", font_size=24, color="#00FFFF")
        # Fix Issue 34: Position label at D4-D5 with scale 0.6
        self.place_in_area(path_label, "D4", "D5", scale_factor=0.6)
        
        self.play(FadeIn(path_label))
        self.wait(2)
