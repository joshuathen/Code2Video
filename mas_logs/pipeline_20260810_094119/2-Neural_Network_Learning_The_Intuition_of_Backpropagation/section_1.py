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
        self.setup_layout("The Learning Goal (The 'Guessing' Game)", [
            "Neural networks are just functions learning to approximate outputs.",
            "The network makes a guess: 'Is this a cat?'",
            "The ground truth says: 'Yes, it is.'",
            "We calculate the error: prediction minus target truth.",
            "Our goal: minimize this error signal."
        ])

        # Assets
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        
        guess = Circle(radius=0.5, color="#FF00FF", fill_opacity=0.5)
        guess_label = Text("Guess 0.7", font_size=20)
        
        target = Circle(radius=0.5, color=WHITE, fill_opacity=0.5)
        target_label = Text("Truth 1.0", font_size=20)
        
        nn_icon = Square(color=BLUE, fill_opacity=0.3).scale(0.5)
        nn_label = Text("NN", font_size=20)
        nn_group = VGroup(nn_icon, nn_label)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(nn_group, 'C2')
        self.place_at_grid(cat_icon, 'C6', scale_factor=0.2)
        self.play(FadeIn(nn_group), FadeIn(cat_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(guess, 'C4', scale_factor=0.9)
        self.place_at_grid(guess_label, 'D4', scale_factor=0.7)
        self.play(FadeIn(guess), FadeIn(guess_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.place_at_grid(target, 'C6', scale_factor=0.9)
        self.place_at_grid(target_label, 'D6', scale_factor=0.7)
        self.play(FadeIn(target), FadeIn(target_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        loss_text = MathTex(r"Loss = |Prediction - Truth|", font_size=28, color=WHITE)
        self.place_at_grid(loss_text, 'B5')
        error_line = Arrow(guess.get_right(), target.get_left(), color=RED)
        error_label = Text("Error 0.3", font_size=20, color=RED).next_to(error_line, UP)
        self.play(Write(loss_text), Create(error_line), Write(error_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        self.play(guess.animate.set_color("#00FF00"), error_line.animate.set_opacity(0.3), error_label.animate.set_opacity(0.3))
        self.play(FadeOut(nn_group), FadeOut(cat_icon), FadeOut(guess), FadeOut(guess_label), FadeOut(target), FadeOut(target_label), FadeOut(loss_text), FadeOut(error_line), FadeOut(error_label))
        self.wait(2)
