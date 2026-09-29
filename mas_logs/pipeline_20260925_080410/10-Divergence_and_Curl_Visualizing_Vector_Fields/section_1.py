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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Vector fields assign vectors to space points.", "Imagine particles flowing in a steady current.", "Arrows represent velocity at every coordinate."]
        self.setup_layout("Introduction: What is a Vector Field?", lecture_lines)
        
        # Define the vector field (PointGrid)
        field = VGroup()
        for x in range(-1, 2):
            for y in range(-1, 2):
                vec = Vector(0.5 * RIGHT + 0.5 * UP, color=BLUE)
                field.add(vec)
        self.place_in_area(field, 'B2', 'F5', scale_factor=0.75)
        
        # Assets
        particle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg")
        current_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/current.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(particle_icon.scale(0.5).move_to(self.grid['C4'])))
        self.play(Create(field), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(field.animate.set_color(BLUE), run_time=1)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        highlighted_vec = field[4].copy()
        highlighted_vec.set_color(YELLOW)
        
        # Place current icon at the highlighted vector
        current_icon.scale(0.3).next_to(highlighted_vec.get_end(), UP, buff=0.1)
        
        self.play(Create(highlighted_vec), FadeIn(current_icon), run_time=0.5)
        self.play(Flash(highlighted_vec.get_end(), color=YELLOW, line_length=0.2), run_time=0.5)
        self.wait(2)
