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
        self.setup_layout("Conclusion: Beyond Luck", [
            "Optimal play is just reducing entropy.", 
            "Information theory makes Wordle a predictable process.", 
            "Math transforms luck into computational strategy."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display the final reduced set of possible words on the screen.svg. (#00FF00)
        screen = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/screen.svg")
        word_list = VGroup(*[Text(w, font_size=24) for w in ["HEART", "PIANO", "ADAPT", "LEAST"]])
        word_list.arrange(DOWN)
        
        self.place_at_grid(screen, 'B5', scale_factor=0.6)
        self.place_in_area(word_list, 'A3', 'C4', scale_factor=0.6)
        
        self.play(FadeIn(screen), Write(word_list))
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.wait(3)

        # === Animation for Lecture Line 2 ===
        # Show the target word being correctly identified by typing on the keyboard.svg. (#FF0000)
        keyboard = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/keyboard.svg")
        target_word = Text("ADAPT", font_size=36, color="#FF0000")
        
        self.place_at_grid(keyboard, 'E5', scale_factor=0.6)
        self.place_at_grid(target_word, 'D6', scale_factor=0.8)
        
        self.play(FadeIn(keyboard), Write(target_word))
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        # Summarize the transition from guessing to strategy. (#FFFFFF)
        final_summary = Text("STRATEGY > LUCK", font_size=30, color=BLUE)
        self.place_at_grid(final_summary, 'E4', scale_factor=0.9)
        
        self.play(FadeIn(final_summary))
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(4)
