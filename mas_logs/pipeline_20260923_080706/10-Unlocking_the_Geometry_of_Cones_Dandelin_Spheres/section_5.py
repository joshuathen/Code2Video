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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Synthesis", ["Conics are 3D geometric results.", "Spheres link visuals to algebra.", "Dandelin spheres unify conic properties."])
        
        # === Animation for Lecture Line 1 ===
        # Show all three conic types appearing from slices. Color: #FFFFFF.
        c1 = Circle(radius=0.5, color=WHITE).set_fill(BLUE, opacity=0.3)
        c2 = Ellipse(width=1.0, height=0.6, color=WHITE).set_fill(RED, opacity=0.3)
        c3 = FunctionGraph(lambda x: 0.5 * x**2, x_range=[-1, 1], color=WHITE).set_fill(GREEN, opacity=0.3).scale(0.5)
        
        self.place_at_grid(c1, 'B2', scale_factor=0.7)
        self.place_at_grid(c2, 'B4', scale_factor=0.7)
        self.place_at_grid(c3, 'B6', scale_factor=0.7)
        
        self.play(FadeIn(c1), FadeIn(c2), FadeIn(c3))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Highlight the common Dandelin sphere principle. Color: #FFFF00.
        sphere1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=YELLOW)
        self.place_at_grid(sphere1, 'E3', scale_factor=0.6)
        
        self.play(FadeIn(sphere1))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Fade all elements into a unified diagram. Color: #FFFFFF.
        summary_text = Text("Dandelin Spheres Unify", font_size=32, color=WHITE)
        self.place_at_grid(summary_text, 'D4')
        
        # Keep sphere but add summary text
        self.play(
            FadeOut(c1), FadeOut(c2), FadeOut(c3),
            FadeIn(summary_text)
        )
        self.lecture[2].set_color("#FFFFFF")
        self.wait(2)
