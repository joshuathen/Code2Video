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
        # Section Title and Lecture Lines
        title = "The Ultimate Leap: Visualizing the 4th Dimension"
        lines = [
            "- A tesseract is the four-dimensional version of a cube.",
            "- We visualize it through 3D projections and cross-sections.",
            "- As it moves, it appears to turn inside out.",
            "- Higher dimensions follow the same logic as lower ones.",
            "- Geometry puzzles help us decode these complex unseen worlds."
        ]
        self.setup_layout(title, lines)

        # Colors
        TESSERACT_COLOR = "#FFFFFF"
        SPHERE_COLOR = "#FFD700"
        HIGHLIGHT_COLOR = "#00FFFF" # Cyan for active line

        # === Animation for Lecture Line 1 ===
        # Display a white wireframe Tesseract (#FFFFFF) as a cube within a cube.
        self.lecture[0].set_color(TESSERACT_COLOR)
        
        # Center the tesseract in the visual area
        center_pos = self.place_in_area(Dot(radius=0, fill_opacity=0), "A1", "F6").get_center()
        
        outer_sq = Square(side_length=3.0, color=TESSERACT_COLOR).move_to(center_pos)
        inner_sq = Square(side_length=1.2, color=TESSERACT_COLOR).move_to(center_pos)
        
        def get_connectors(o_sq, i_sq):
            return VGroup(*[
                Line(o_sq.get_vertices()[i], i_sq.get_vertices()[i], color=TESSERACT_COLOR)
                for i in range(4)
            ])
        
        connectors = get_connectors(outer_sq, inner_sq)
        tesseract = VGroup(outer_sq, inner_sq, connectors)
        
        self.play(Create(tesseract), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We visualize it through 3D projections and cross-sections.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        projection_label = Text("3D Projection", font_size=20, color=WHITE)
        # Fix Issue 34: Move to area A2-A5 to prevent cutoff
        self.place_in_area(projection_label, 'A2', 'A5')
        self.play(Write(projection_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # As it moves, it appears to turn inside out.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Inside-out rotation/scaling animation using ValueTracker for persistence
        ratio = ValueTracker(0)
        
        def update_tesseract(mob):
            r = ratio.get_value()
            # Interpolate sizes to simulate "turning inside out"
            s_outer = 3.0 * (1 - r) + 1.2 * r
            s_inner = 1.2 * (1 - r) + 3.0 * r
            
            # Rotation to add depth sensation
            angle = r * PI/2
            
            # Efficiently update the components using become
            mob[0].become(Square(side_length=s_outer, color=TESSERACT_COLOR).move_to(center_pos).rotate(angle))
            mob[1].become(Square(side_length=s_inner, color=TESSERACT_COLOR).move_to(center_pos).rotate(-angle))
            mob[2].become(get_connectors(mob[0], mob[1]))

        tesseract.add_updater(update_tesseract)
        self.play(ratio.animate.set_value(1), run_time=3, rate_func=there_and_back)
        tesseract.remove_updater(update_tesseract)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Higher dimensions follow the same logic as lower ones. (Sphere through plane)
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(SPHERE_COLOR)
        
        self.play(FadeOut(tesseract), FadeOut(projection_label))
        
        # 2D Plane (Horizontal)
        # Fix Issue 36: Move to area C1-C5 for better visibility
        plane_line = Line(LEFT*2.0, RIGHT*2.0, color=WHITE)
        self.place_in_area(plane_line, 'C1', 'C5') 
        
        plane_label = Text("2D Plane", font_size=18, color=WHITE)
        self.place_at_grid(plane_label, "C6")
        
        # 3D Sphere represented as a moving circle
        sphere_obj = Circle(radius=1.0, color=SPHERE_COLOR, stroke_opacity=0.5, fill_opacity=0.2)
        sphere_obj.move_to(plane_line.get_center() + UP * 2.0)
        
        # The cross-section slice that appears on the plane
        slice_circle = Circle(radius=0.01, color=SPHERE_COLOR, fill_opacity=0.8)
        slice_circle.move_to(plane_line.get_center())
        
        self.play(Create(plane_line), Write(plane_label), FadeIn(sphere_obj))
        
        # Animation: Sphere moves down, slice grows and shrinks (3D to 2D analogy)
        def slice_updater(mob):
            # Vertical distance from sphere center to plane line
            d = abs(sphere_obj.get_center()[1] - plane_line.get_center()[1])
            R = 1.0 # Sphere radius
            if d < R:
                # r_slice = sqrt(R^2 - d^2)
                r_slice = np.sqrt(R**2 - d**2)
                mob.set_width(max(0.01, 2 * r_slice))
                mob.set_opacity(0.8)
            else:
                mob.set_opacity(0)

        slice_circle.add_updater(slice_updater)
        self.add(slice_circle)
        
        self.play(sphere_obj.animate.shift(DOWN * 4.0), run_time=4)
        slice_circle.remove_updater(slice_updater)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Geometry puzzles help us decode these complex unseen worlds.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        analogy_text = Text("4D Object -> 3D Shadow", font_size=24, color=WHITE)
        # Fix Issue 35: Move to area F2-F5 to avoid overlap with sphere animation
        self.place_in_area(analogy_text, 'F2', 'F5', scale_factor=0.8)
        
        self.play(Write(analogy_text))
        self.wait(3)

        # Cleanup
        self.play(FadeOut(VGroup(sphere_obj, plane_line, plane_label, slice_circle, analogy_text)))
