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
        self.setup_layout("Defining Duality: The Linear Functional", [
            "A fixed vector defines a linear functional.",
            "It maps input vectors to scalar outputs.",
            "This machine acts as a weight filter."
        ])
        
        # Elements
        w_vec = Vector([1, 1], color=YELLOW)
        v_vec = Vector([1.5, 0.5], color=BLUE)
        f_label = MathTex("f(v) = w \\cdot v", color=WHITE)
        filter_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg", color=WHITE)
        
        # Labels
        label_w = Tex("w", color=YELLOW)
        label_v = Tex("v", color=BLUE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(w_vec, 'C4', scale_factor=0.9)
        self.place_at_grid(filter_icon, 'B4', scale_factor=0.6)
        self.add(w_vec, filter_icon)
        self.place_at_grid(label_w, 'B3') # next to w_vec location
        self.add(label_w)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        self.place_at_grid(v_vec, 'D4', scale_factor=0.9)
        self.play(Create(v_vec))
        self.place_at_grid(label_v, 'D5')
        self.play(Write(label_v))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF33A8")
        self.place_in_area(f_label, 'C5', 'D6', scale_factor=0.8)
        self.play(Write(f_label))
        
        # Draw projection line
        line = Line(v_vec.get_end(), w_vec.get_end(), color="#FF33A8", stroke_width=2)
        self.play(Create(line))
        self.wait(3)
