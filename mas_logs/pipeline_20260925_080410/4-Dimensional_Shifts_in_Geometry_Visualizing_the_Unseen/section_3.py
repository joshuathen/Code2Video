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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["The Slice Method reveals hidden internal geometry.", "Moving an object through space changes its cross-section.", "Observe the changing area during rotation."]
        self.setup_layout("Cross-Sectional Dynamics", lecture_lines)
        
        # Animation setup
        # Use SVG asset as requested in Issue 18
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        sphere_label = Text("Sphere", font_size=20, color=WHITE)
        self.place_at_grid(sphere_label, 'A4')
        self.place_in_area(sphere, 'B4', 'B6', scale_factor=0.5)
        self.play(FadeIn(sphere), Write(sphere_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        cross_section = Circle(radius=0, color="#00FF00", fill_opacity=0.6)
        cs_label = Text("Cross-section", font_size=20, color="#00FF00")
        
        self.place_at_grid(cs_label, 'D4', scale_factor=0.8)
        self.place_in_area(cross_section, 'E4', 'E6', scale_factor=0.6)
        
        self.play(
            sphere.animate.shift(DOWN * 2),
            FadeIn(cross_section),
            Write(cs_label)
        )
        self.play(cross_section.animate.set_width(2.0))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        exit_label = Text("Exit", font_size=20, color="#FF00FF")
        self.place_at_grid(exit_label, 'D6', scale_factor=0.8)
        
        self.play(
            sphere.animate.shift(DOWN * 2),
            cross_section.animate.set_width(0.0),
            FadeOut(cross_section),
            Write(exit_label)
        )
        self.wait(1)
