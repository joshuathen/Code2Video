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
        self.setup_layout("The Intuition: Contextual Awareness", 
                          ["Attention is all about selective focus.", 
                           "Words need context to be understood.", 
                           "The model tracks relations between words."])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.lecture[0].set_color("#FFFFFF")
        
        context_text = Text("Context", font_size=48, color="#FFFFFF")
        # Fixed: issue #20/#35
        self.place_at_grid(context_text, 'B2', scale_factor=0.9)
        self.play(FadeIn(context_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        self.lecture[1].set_color("#FF00FF")
        
        # Fixed: Integrate asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # Assuming none.svg is a vector icon representing the word vector.
        word_vector_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color="#FF00FF")
        self.place_at_grid(word_vector_icon, 'C4', scale_factor=0.75) # Fixed: Issue #22/#37
        
        self.play(FadeIn(word_vector_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        self.lecture[2].set_color("#00FFFF")
        
        heatmap = Rectangle(width=2, height=1, color="#00FFFF", fill_opacity=0.3)
        # Fixed: issue #21/#36
        self.place_in_area(heatmap, 'C3', 'E4', scale_factor=0.8)
        
        self.play(FadeIn(heatmap))
        self.wait(2)
