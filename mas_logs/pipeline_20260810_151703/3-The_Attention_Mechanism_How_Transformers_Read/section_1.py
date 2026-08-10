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
        self.setup_layout("The Core Problem: Contextual Ambiguity", [
            "Previous models processed words only linearly.", 
            "This misses dependencies across long distances.", 
            "Ambiguity arises when words shift meaning."
        ])
        
        # Assets
        dictionary_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dictionary.svg")
        pen_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pen.svg")
        
        # === Animation for Lecture Line 1 ===
        # Previous models processed words only linearly.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create 'Word Ambiguity' with assets
        ambiguity_text = Text("Word Ambiguity", font_size=24, color=WHITE)
        ambiguity_group = VGroup(ambiguity_text, dictionary_icon).arrange(RIGHT, buff=0.2)
        self.place_in_area(ambiguity_group, "A1", "A3", scale_factor=0.7)
        self.play(Write(ambiguity_text), FadeIn(dictionary_icon))
        
        # Visualize \"Linear Processing\"
        words = VGroup(*[Text(w, font_size=24) for w in ["The", "river", "bank"]])
        words.arrange(RIGHT, buff=0.5)
        self.place_in_area(words, "B1", "B6")
        self.play(Write(words))
        
        # Arrows showing sequential flow
        arrows = VGroup(*[Arrow(words[i].get_right(), words[i+1].get_left(), buff=0.1, color=BLUE) for i in range(2)])
        self.play(Create(arrows))
        
        # === Animation for Lecture Line 2 ===
        # This misses dependencies across long distances.
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        # Show long distance dependency
        long_arrow = CurvedArrow(words[0].get_top(), words[2].get_top(), angle=-PI/2, color=RED)
        
        # Fix #20: Move label position
        missed_context = Text("Missed Context", font_size=18, color=RED)
        self.place_at_grid(missed_context, "C3", scale_factor=0.9)
        
        # Include pen icon in analysis process
        self.place_at_grid(pen_icon, "C5", scale_factor=0.8)
        
        self.play(Create(long_arrow), Write(missed_context), FadeIn(pen_icon))
        
        # === Animation for Lecture Line 3 ===
        # Ambiguity arises when words shift meaning.
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        # Highlight \"bank\"
        bank_word = words[2]
        self.play(bank_word.animate.set_color(GOLD), Indicate(bank_word))
        
        # Fix #21: Move footer position
        comparison_label = Text("river bank vs bank account", font_size=20, color="#00FFFF")
        self.place_in_area(comparison_label, "E2", "E5", scale_factor=0.75)
        self.play(FadeIn(comparison_label))
        
        self.wait(2)
