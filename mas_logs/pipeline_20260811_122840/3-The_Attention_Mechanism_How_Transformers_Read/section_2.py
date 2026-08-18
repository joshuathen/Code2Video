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
        lecture_lines = ["Query asks what I need.", "Key acts as a label.", "Value provides the actual content."]
        self.setup_layout("The Core Components: Q, K, and V", lecture_lines)
        
        # Define components
        q_box = Square(color="#FF5733", fill_opacity=0.5)
        k_box = Square(color="#33FF57", fill_opacity=0.5)
        v_box = Square(color="#3357FF", fill_opacity=0.5)
        
        q_label = Text("Q", font_size=24, color="#FF5733").next_to(q_box, UP, buff=0.1)
        k_label = Text("K", font_size=24, color="#33FF57").next_to(k_box, UP, buff=0.1)
        v_label = Text("V", font_size=24, color="#3357FF").next_to(v_box, UP, buff=0.1)
        
        # Load asset
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg").set_color(WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        # Placing components in C-column row segment as per instructions
        self.place_at_grid(VGroup(q_box, q_label), "C2", scale_factor=0.6)
        self.place_at_grid(VGroup(k_box, k_label), "C4", scale_factor=0.6)
        self.place_at_grid(VGroup(v_box, v_label), "C6", scale_factor=0.6)
        self.play(Create(VGroup(q_box, q_label, k_box, k_label, v_box, v_label)))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"))
        self.place_at_grid(magnifier, "C3", scale_factor=0.8)
        self.play(FadeIn(magnifier))
        # Move across K to find match
        self.play(magnifier.animate.move_to(self.grid["C4"]), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#3357FF"))
        v_final = v_box.copy().set_fill(opacity=1)
        self.place_at_grid(v_final, "D4", scale_factor=0.9)
        self.play(Transform(v_box.copy(), v_final))
        self.wait(2)
