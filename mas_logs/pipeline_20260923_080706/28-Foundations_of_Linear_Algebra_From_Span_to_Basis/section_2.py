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
        self.setup_layout("Linear Combinations and Span", [
            "Linear combinations sum multiple scaled vectors.", 
            "Span is the set of all reachable points.", 
            "Two non-collinear vectors span the whole plane."
        ])
        
        # Load Assets
        grid_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        ruler_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        protractor_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        
        # Grid Title
        grid_title = Text("Grid Visual", font_size=24, color=WHITE)
        self.place_at_grid(grid_title, 'A4')
        
        # Define vectors
        origin = self.grid["E2"]
        v = Vector([1, 1], color="#33FF57")
        w = Vector([2, -1], color="#33FF57")
        v.shift(origin); w.shift(origin)
        vector_group = VGroup(v, w)
        self.place_at_grid(vector_group, 'E2', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_at_grid(grid_img, 'C3', scale_factor=0.5)
        self.play(FadeIn(grid_img), Create(vector_group))
        self.lecture[0].set_color("#33FF57")
        
        # === Animation for Lecture Line 2 ===
        self.place_in_area(self.lecture[1], 'A1', 'D1', scale_factor=0.7)
        self.play(FadeIn(self.lecture[1]))
        
        scalar = ValueTracker(1.0)
        v_scaled = Vector([1, 1], color="#3357FF")
        v_scaled.add_updater(lambda m: m.become(Vector([scalar.get_value() * 1, scalar.get_value() * 1], color="#3357FF").move_to(origin)))
        self.add(v_scaled)
        
        self.place_at_grid(ruler_img, 'E4', scale_factor=0.4)
        self.play(FadeIn(ruler_img), scalar.animate.set_value(2.0), run_time=2)
        self.lecture[1].set_color("#3357FF")
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        
        v_plus_w = Vector([3, 0], color=WHITE).shift(origin)
        self.place_at_grid(protractor_img, 'D5', scale_factor=0.4)
        self.play(FadeIn(protractor_img), Create(v_plus_w))
        self.lecture[2].set_color(WHITE)
        self.wait(2)
