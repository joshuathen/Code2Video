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
        # Data setup
        title = "The Mystery of the Flat Shadow"
        lecture_lines = [
            "Start with a sphere, catching parallel rays of light.",
            "These parallel beams move steadily toward the sphere's surface.",
            "On the other side, a flat grey shadow appears.",
            "Both the sphere and shadow share the same radius.",
            "Even as the sphere rotates, the shadow stays constant."
        ]
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Start with a sphere, catching parallel rays of light.
        self.lecture[0].set_color("#FFFF00")
        
        # Sphere: white circle with radial gradient effect
        # Radius 0.45 fits well within a 1x1 grid cell
        sphere = Circle(radius=0.45, color=WHITE, fill_opacity=1)
        sphere.set_fill(color=[WHITE, "#555555"])
        sphere.set_stroke(width=0)
        self.place_at_grid(sphere, 'C3')
        
        # Parallel yellow rays starting at column 1
        light_lines = VGroup()
        for i in range(5):
            y_shift = (i - 2) * 0.2
            line = Line(
                start=self.grid['C1'] + UP*y_shift, 
                end=self.grid['C1'] + RIGHT*0.7 + UP*y_shift, 
                color="#FFFF00", 
                stroke_width=2
            )
            light_lines.add(line)
            
        self.play(FadeIn(sphere))
        self.play(Create(light_lines))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # These parallel beams move steadily toward the sphere's surface.
        self.lecture[1].set_color("#FFFF00")
        
        # Move beams from C1 area toward C3
        self.play(
            light_lines.animate.shift(RIGHT * 1.5),
            run_time=2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # On the other side, a flat grey shadow appears.
        self.lecture[2].set_color("#808080")
        
        # Grey circular shadow at C6
        shadow = Circle(radius=0.45, color="#808080", fill_opacity=0.5)
        shadow.set_stroke(color="#808080", width=1)
        self.place_at_grid(shadow, 'C6')
        
        # Projection lines (rays extending from sphere C3 to shadow C6)
        projection_rays = VGroup()
        for i in range(5):
            y_shift = (i - 2) * 0.2
            ray = Line(
                start=self.grid['C3'] + RIGHT*0.45 + UP*y_shift,
                end=self.grid['C6'] + LEFT*0.45 + UP*y_shift,
                color="#FFFF00",
                stroke_opacity=0.3
            )
            projection_rays.add(ray)
            
        self.play(Create(projection_rays))
        self.play(FadeIn(shadow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Both the sphere and shadow share the same radius.
        self.lecture[3].set_color("#00FFFF")
        
        # Cyan radius lines
        sphere_radius = Line(sphere.get_center(), sphere.get_center() + RIGHT*0.45, color="#00FFFF")
        shadow_radius = Line(shadow.get_center(), shadow.get_center() + RIGHT*0.45, color="#00FFFF")
        
        r_label_1 = Text("r", font_size=18, color="#00FFFF")
        r_label_2 = Text("r", font_size=18, color="#00FFFF")
        r_label_1.next_to(sphere_radius, UP, buff=0.1)
        r_label_2.next_to(shadow_radius, UP, buff=0.1)
        
        self.play(
            Create(sphere_radius), 
            Create(shadow_radius),
            Write(r_label_1),
            Write(r_label_2)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Even as the sphere rotates, the shadow stays constant.
        self.lecture[4].set_color("#FFFFFF")
        
        # Add visual meridian mark for rotation
        meridian = Ellipse(width=0.1, height=0.9, color=WHITE, stroke_opacity=0.4).move_to(sphere)
        self.add(meridian)
        
        # Group sphere elements for rotation
        sphere_system = VGroup(sphere, meridian, sphere_radius, r_label_1)
        
        self.play(
            Rotate(sphere_system, angle=2*PI, about_point=sphere.get_center()),
            run_time=4,
            rate_func=linear
        )
        self.wait(2)
