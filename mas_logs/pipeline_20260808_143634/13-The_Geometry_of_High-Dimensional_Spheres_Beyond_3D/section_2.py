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
        lecture_lines = ["As dimensions increase, sphere volume shrinks to zero.", "Surprisingly, volume migrates to the thin outer shell.", "Imagine an orange with almost no pulp."]
        self.setup_layout("The Counter-Intuitive 'Spiky' Sphere", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF0000")
        circle = Circle(radius=1.5, color=WHITE, fill_opacity=0.3)
        self.place_at_grid(circle, 'C2')
        vol_label = Text("Volume -> 0", color="#FF0000", font_size=24)
        self.place_at_grid(vol_label, 'A2')
        self.play(Create(circle), Write(vol_label))
        self.play(circle.animate.scale(0.2), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        shell = Annulus(inner_radius=1.2, outer_radius=1.5, color="#00FFFF", fill_opacity=0.8)
        self.place_at_grid(shell, 'C2')
        self.play(FadeIn(shell))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFA500")
        
        orange = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg")
        self.place_at_grid(orange, 'D5')
        pulp_label = Text("Pulp", color="#FFA500", font_size=24)
        self.place_at_grid(pulp_label, 'D3')
        
        self.play(FadeIn(orange), Write(pulp_label))
        self.play(orange.animate.scale(0.3), run_time=2)
        
        self.wait(2)
