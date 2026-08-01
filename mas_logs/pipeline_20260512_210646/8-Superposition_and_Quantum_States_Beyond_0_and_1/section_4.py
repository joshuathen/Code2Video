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
        # Updated lecture lines
        lecture_lines = [
            "We map quantum states onto a sphere's surface.",
            "Zero is at the top; one is below.",
            "Every point on the sphere is a superposition."
        ]
        
        self.setup_layout("Visualizing the State: The Bloch Sphere", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Color line 1 steel blue
        self.lecture[0].set_color("#4682B4")
        
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/sphere.svg
        # Loading and placing the Bloch Sphere asset
        bloch_sphere = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/sphere.svg")
        bloch_sphere.set_color("#4682B4")
        # Place in area spanning rows B-E and columns 2-5
        self.place_in_area(bloch_sphere, "B2", "E5", scale_factor=1.5)
        sphere_center = bloch_sphere.get_center()
        sphere_radius = bloch_sphere.height / 2
        
        # Dashed ellipse equator to simulate 3D on 2D
        equator = Ellipse(width=sphere_radius * 2, height=sphere_radius * 0.4, color="#4682B4", stroke_width=2)
        equator = DashedVMobject(equator)
        equator.move_to(sphere_center)
        
        self.play(
            DrawBorderThenFill(bloch_sphere),
            Create(equator),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        
        # Labels for poles
        # North Pole |0⟩
        label_0 = Text("|0⟩", color="#FF0000", font_size=36)
        # Position centered horizontally above the sphere
        self.place_in_area(label_0, "A3", "A4", scale_factor=1.0)
        
        # South Pole |1⟩
        label_1 = Text("|1⟩", color="#00FF00", font_size=36)
        # Position centered horizontally below the sphere
        self.place_in_area(label_1, "F3", "F4", scale_factor=1.0)

        self.play(
            FadeIn(label_0),
            FadeIn(label_1),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color line 3 yellow
        self.lecture[2].set_color("#FFFF00")
        
        # Yellow vector arrow rotating from center
        # Starting pointing to |0⟩
        vector = Arrow(
            start=sphere_center,
            end=sphere_center + UP * sphere_radius,
            buff=0,
            color="#FFFF00",
            stroke_width=6,
            tip_length=0.25
        )
        
        self.play(GrowArrow(vector))
        self.wait(0.5)
        
        # Rotation 1: To an arbitrary superposition point
        angle1 = -PI / 3
        rot_point1 = sphere_center + np.array([sphere_radius * np.sin(angle1), sphere_radius * np.cos(angle1), 0])
        
        self.play(
            vector.animate.put_start_and_end_on(sphere_center, rot_point1),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Rotation 2: To the equator
        angle2 = -PI / 2
        rot_point2 = sphere_center + np.array([sphere_radius * np.sin(angle2), 0.1, 0])
        
        self.play(
            vector.animate.put_start_and_end_on(sphere_center, rot_point2),
            run_time=1.5
        )
        self.wait(1)
        
        # Final rotation to another point on the surface
        angle3 = -3 * PI / 4
        rot_point3 = sphere_center + np.array([sphere_radius * np.sin(angle3), sphere_radius * np.cos(angle3), 0])
        
        self.play(
            vector.animate.put_start_and_end_on(sphere_center, rot_point3),
            run_time=1.5
        )
        self.wait(2)
