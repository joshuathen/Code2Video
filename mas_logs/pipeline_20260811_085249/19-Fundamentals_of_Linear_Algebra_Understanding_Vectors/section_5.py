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
        lecture_lines = ["Linear algebra is the language of motion.", "It powers computer graphics and physics.", "Models are just collections of many vectors."]
        self.setup_layout("Conclusion and Application Summary", lecture_lines)
        
        # Vectors
        u = Vector([1, 1], color=BLUE)
        v = Vector([2, -0.5], color=YELLOW)
        u_plus_v = Vector([3, 0.5], color=RED)
        
        # Icons
        physics_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/physics.svg")
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        
        # Labels
        u_label = Text("u", font_size=16)
        v_label = Text("v", font_size=16)
        uv_label = Text("u+v", font_size=16)
        labels = VGroup(u_label, v_label, uv_label).arrange(RIGHT)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_at_grid(physics_icon, 'A3', scale_factor=0.3)
        self.place_at_grid(u, 'B3', scale_factor=0.8)
        self.place_at_grid(v, 'B4', scale_factor=0.8)
        self.place_at_grid(u_plus_v, 'C4', scale_factor=0.8)
        self.place_at_grid(labels, 'C4', scale_factor=0.7)
        self.play(FadeIn(physics_icon), FadeIn(u), FadeIn(v), FadeIn(u_plus_v), Write(labels))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        area = Polygon(self.grid['B2'], self.grid['B5'], self.grid['E5'], self.grid['E2'], color=YELLOW, fill_opacity=0.3)
        self.play(Create(area))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        summary_box = Rectangle(width=4, height=2, color=WHITE)
        summary_text = Text("Linear Algebra = \nLanguage of Motion", font_size=20)
        summary = VGroup(summary_box, summary_text, computer_icon).arrange(DOWN)
        self.place_in_area(summary, 'D2', 'F5', scale_factor=0.7)
        self.play(Write(summary))
        self.wait(2)
