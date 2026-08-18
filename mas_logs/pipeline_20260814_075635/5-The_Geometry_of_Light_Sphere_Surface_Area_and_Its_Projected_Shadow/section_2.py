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
        lecture_lines = [
            "The sphere surface area formula is 4 pi r squared.",
            "Imagine unwrapping the sphere's surface.",
            "The surface covers exactly four flat circles.",
            "Each circle has the sphere's radius.",
            "Total area is four times pi r squared."
        ]
        self.setup_layout("Sphere Surface", lecture_lines)
        
        # Asset usage
        sphere_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        sphere_icon.set_color("#00FFFF")
        # Apply layout fixes from issue 20 and 22
        self.place_at_grid(sphere_icon, 'D5', scale_factor=0.7)
        
        # Point P
        dot = Dot(color="#FF0000")
        dot.move_to(sphere_icon.get_top())
        label_p = Text("P", font_size=20, color="#FF0000").next_to(dot, UP)
        
        # Formula group
        sphere_formula = MathTex("A = 4\\pi r^2", color=WHITE)
        self.place_in_area(sphere_formula, 'A2', 'C4', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(sphere_icon), FadeIn(dot), FadeIn(label_p), FadeIn(sphere_formula))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        self.play(sphere_icon.animate.rotate(2*PI), run_time=2)
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#0000FF"))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFA500"))
        self.wait(2)
