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
        self.setup_layout("Summary and Conclusion", [
            "Fourier Transform bridges time and frequency.",
            "Essential for modern signal processing.",
            "Powering MP3s and medical imaging."
        ])
        
        # Setup visual elements
        time_domain = Text("Time Domain", color=BLUE)
        freq_domain = Text("Frequency Domain", color=YELLOW)
        
        # Assets
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg", color=WHITE)
        mp3_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mp3.svg", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # Reiterate the core concepts: Time and Frequency domains.
        self.lecture[0].set_color("#FFFF00")
        
        self.place_at_grid(time_domain, 'B2', scale_factor=0.6)
        self.place_at_grid(prism, 'B3', scale_factor=0.6)
        self.place_at_grid(freq_domain, 'B4', scale_factor=0.6)
        self.play(FadeIn(time_domain), FadeIn(prism), FadeIn(freq_domain))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Summarize the prism analogy as a filter tool.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        label = Text("Signal Processing", font_size=30, color=WHITE)
        self.place_in_area(label, 'C2', 'C4', scale_factor=0.7)
        self.play(Write(label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display key terms: 'Fourier Analysis' on screen with the audio representation.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        
        final_text = Text("Fourier Analysis", font_size=40, color=WHITE)
        self.place_in_area(final_text, 'D2', 'D4', scale_factor=0.8)
        self.place_at_grid(mp3_icon, 'E3', scale_factor=0.5)
        
        self.play(FadeIn(final_text), FadeIn(mp3_icon), run_time=2)
        self.wait(2)
