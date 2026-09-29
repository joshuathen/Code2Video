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
        self.setup_layout("Physical Application: Torque", [
            "Torque defines rotational force effect.",
            "Cross product relates radius and force.",
            "Angle determines the swing speed."
        ])
        
        # Assets
        bolt = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bolt.svg")
        wrench = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wrench.svg")
        
        self.place_at_grid(bolt, 'D4', scale_factor=0.3)
        self.place_at_grid(wrench, 'D5', scale_factor=0.3)
        
        # Define objects
        pivot = bolt
        r_vec = Vector([1.5, 0, 0], color=BLUE)
        f_vec = Vector([0, 1.2, 0], color=RED)
        
        # Position using grid constraints (Issues 32, 47)
        self.place_at_grid(pivot, 'D4', scale_factor=0.4)
        r_vec.move_to(pivot.get_center() + [0.75, 0, 0])
        f_vec.move_to(pivot.get_center() + [1.5, 0.6, 0])
        
        r_label = MathTex(r"\vec{r}", color=BLUE)
        f_label = MathTex(r"\vec{F}", color=RED)
        
        # Position labels (Issues 30, 45)
        self.place_at_grid(r_label, 'D5', scale_factor=0.9)
        self.place_at_grid(f_label, 'C5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(pivot), Create(r_vec), Write(r_label))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(Create(f_vec), Write(f_label))
        self.lecture[1].set_color(BLUE)

        # === Animation for Lecture Line 3 ===
        torque_label = MathTex(r"\vec{\tau} = \vec{r} \times \vec{F}", color=GREEN)
        # Position torque formula (Issues 31, 46)
        self.place_at_grid(torque_label, 'B4', scale_factor=1.0)
        self.play(Write(torque_label), FadeIn(wrench))
        self.lecture[2].set_color(RED)
        
        self.wait(2)
