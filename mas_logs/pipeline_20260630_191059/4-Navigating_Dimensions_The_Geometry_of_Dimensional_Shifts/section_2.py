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

class Section2Scene(TeachingScene):
    def construct(self):
        # DATA
        lecture_lines = [
            "Higher dimensions appear as shifting slices in lower worlds.",
            "Imagine a sphere passing through a flat two-dimensional plane.",
            "First, a tiny point appears on the surface.",
            "The point grows into a wide, expanding circle.",
            "Finally, the circle shrinks and vanishes into nothing."
        ]
        
        self.setup_layout("The Flatland Perspective: Slices and Shadows", lecture_lines)
        
        # Colors
        COLOR_2D_WORLD = "#FFFFFF"
        COLOR_SPHERE = "#00FF00"
        COLOR_DIM = "#444444"
        
        # === Animation for Lecture Line 1 ===
        # Higher dimensions appear as shifting slices in lower worlds.
        self.play(self.lecture[0].animate.set_color(COLOR_2D_WORLD))
        
        # Draw a horizontal white line across the screen labeled '2D World'
        world_line = Line(self.grid["D1"], self.grid["D6"], color=COLOR_2D_WORLD)
        world_label = Text("2D World", font_size=18, color=COLOR_2D_WORLD)
        # Resolved Issue 28: Repositioning world_label to E3
        self.place_at_grid(world_label, "E3", scale_factor=0.8)
        
        self.play(Create(world_line), Write(world_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Imagine a sphere passing through a flat two-dimensional plane.
        self.play(
            self.lecture[0].animate.set_color(COLOR_DIM),
            self.lecture[1].animate.set_color(COLOR_SPHERE)
        )
        
        # A 3D sphere (represented by a circle with shading) moves vertically towards the line
        sphere_radius = 1.0
        sphere = Circle(radius=sphere_radius, color=COLOR_SPHERE, fill_opacity=0.3)
        self.place_at_grid(sphere, "B3")
        
        self.play(FadeIn(sphere))
        self.wait(1)

        # ValueTracker for Sphere Y-pos movement
        # B3 y is 1.2. D3 y is -0.8.
        sphere_pos = ValueTracker(self.grid["B3"][1])
        sphere.add_updater(lambda m: m.move_to([self.grid["B3"][0], sphere_pos.get_value(), 0]))
        
        # === Animation for Lecture Line 3 ===
        # First, a tiny point appears on the surface.
        self.play(
            self.lecture[1].animate.set_color(COLOR_DIM),
            self.lecture[2].animate.set_color(COLOR_SPHERE)
        )
        
        # Intersection logic for the 1D slice (line segment on the 2D world line)
        world_y = self.grid["D3"][1]
        slice_line = Line(color=COLOR_SPHERE, stroke_width=6)
        
        def update_slice(m):
            y = sphere_pos.get_value()
            dist = abs(y - world_y)
            if dist < sphere_radius:
                w = np.sqrt(max(0, sphere_radius**2 - dist**2))
                m.set_points_as_corners([
                    [self.grid["D3"][0] - w, world_y, 0],
                    [self.grid["D3"][0] + w, world_y, 0]
                ])
                m.set_stroke(opacity=1)
            else:
                # Handle the exact contact point
                if abs(dist - sphere_radius) < 0.05:
                     m.set_points_as_corners([
                        [self.grid["D3"][0] - 0.05, world_y, 0],
                        [self.grid["D3"][0] + 0.05, world_y, 0]
                    ])
                     m.set_stroke(opacity=1)
                else:
                    m.set_stroke(opacity=0)
        
        slice_line.add_updater(update_slice)
        self.add(slice_line)
        
        # Resolved Issue 26: Repositioning contact_label to area B4-B5
        contact_label = Text("Point of Contact", font_size=18, color=COLOR_SPHERE)
        self.place_in_area(contact_label, 'B4', 'B5', scale_factor=0.8)
        
        # Move sphere to touch point (Center at Row C means bottom touches Row D)
        self.play(sphere_pos.animate.set_value(self.grid["C3"][1]), run_time=1.5)
        self.play(Write(contact_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The point grows into a wide, expanding circle.
        # Note: In this edge-on view, the circle appears as an expanding 1D slice (segment).
        self.play(
            self.lecture[2].animate.set_color(COLOR_DIM),
            self.lecture[3].animate.set_color(COLOR_SPHERE)
        )
        
        # Resolved Issue 27: Repositioning slice_label to C4
        slice_label = Text("1D Slice", font_size=18, color=COLOR_SPHERE)
        self.place_at_grid(slice_label, 'C4', scale_factor=0.8)
        
        # Move sphere to halfway through (Center at Row D)
        self.play(
            FadeOut(contact_label),
            FadeIn(slice_label),
            sphere_pos.animate.set_value(self.grid["D3"][1]), 
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Finally, the circle shrinks and vanishes into nothing.
        self.play(
            self.lecture[3].animate.set_color(COLOR_DIM),
            self.lecture[4].animate.set_color(COLOR_SPHERE)
        )
        
        # Move sphere past the line (Center at Row E means top touches Row D)
        self.play(sphere_pos.animate.set_value(self.grid["E3"][1]), run_time=2)
        
        # Move to Row F to fully exit
        self.play(
            sphere_pos.animate.set_value(self.grid["F3"][1]),
            FadeOut(slice_label),
            FadeOut(sphere),
            run_time=1.5
        )
        self.wait(1)
        
        # Finish
        self.play(self.lecture[4].animate.set_color(COLOR_DIM))
        self.wait(2)
