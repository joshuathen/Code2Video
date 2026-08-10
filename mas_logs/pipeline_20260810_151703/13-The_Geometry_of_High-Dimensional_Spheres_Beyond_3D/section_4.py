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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["We visualize hyperspheres using shadows and slices.", "Projections help us perceive four-dimensional objects.", "Light projects 4D shadow onto 2D."]
        self.setup_layout("Visualization Techniques: Projection and Slicing", lecture_lines)
        
        # Create objects
        circle_4d_shadow = Circle(radius=0.8, color=BLUE)
        dot_center = Dot(color=YELLOW)
        projection_screen = Line(start=[-1.0, 0, 0], end=[1.0, 0, 0], color=GRAY)
        light_source = Dot(point=[0, 0, 0], color=WHITE)
        
        # Group assets for easier positioning
        animation_group = VGroup(circle_4d_shadow, dot_center, projection_screen, light_source)
        
        # Apply positioning constraints from issues 30, 31, 32
        self.place_at_grid(light_source, 'B5', scale_factor=0.5)
        self.place_at_grid(circle_4d_shadow, 'D5', scale_factor=0.5)
        self.place_at_grid(projection_screen, 'F5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(circle_4d_shadow))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_at_grid(dot_center, 'D5', scale_factor=0.8)
        self.play(FadeIn(dot_center))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(Create(projection_screen), FadeIn(light_source))
        
        # Pulsing shadow simulation
        shadow_path = Circle(radius=0.4, color=BLUE)
        shadow_path.move_to(self.grid['D5'])
        self.play(Transform(circle_4d_shadow, shadow_path))
        self.wait(1)
