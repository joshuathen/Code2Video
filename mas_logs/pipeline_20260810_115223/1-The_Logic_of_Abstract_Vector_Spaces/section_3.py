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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application: The Library of Sounds", [
            "Audio signals can be vectors.", 
            "Each time point is a dimension.", 
            "Adding clips creates a new sound.", 
            "This represents signal superposition.", 
            "Abstraction simplifies complex sound analysis."
        ])
        
        # Load Library Icon
        library_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg")
        
        # === Animation for Lecture Line 1 ===
        # Audio signals can be vectors.
        self.lecture[0].set_color("#FF5733")
        self.place_at_grid(library_icon, 'C1', scale_factor=0.5)
        self.play(FadeIn(library_icon))
        
        wave = Line(LEFT, RIGHT, color="#FF5733").set_length(2)
        self.place_at_grid(wave, 'C2', scale_factor=0.6)
        self.play(Create(wave))

        # === Animation for Lecture Line 2 ===
        # Each time point is a dimension.
        self.lecture[1].set_color("#FF5733")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Adding clips creates a new sound.
        self.lecture[2].set_color("#FF33FF")
        bass = Arrow(ORIGIN, UP*1.0, color="#33FF57", buff=0)
        treble = Arrow(ORIGIN, RIGHT*1.0, color="#3357FF", buff=0)
        
        bass_label = Text("Bass", font_size=16, color="#33FF57")
        treble_label = Text("Treble", font_size=16, color="#3357FF")
        
        bass_group = VGroup(bass, bass_label).arrange(UP)
        treble_group = VGroup(treble, treble_label).arrange(RIGHT)
        
        self.place_in_area(bass_group, 'B4', 'B5', scale_factor=0.7)
        self.place_in_area(treble_group, 'C4', 'C5', scale_factor=0.7)
        self.play(Create(bass_group), Create(treble_group))

        # === Animation for Lecture Line 4 ===
        # This represents signal superposition.
        self.lecture[3].set_color("#FF33FF")
        chord = Arrow(ORIGIN, (UP+RIGHT)*1.0, color="#FF33FF", buff=0)
        chord_label = Text("Chord", font_size=16, color="#FF33FF")
        chord_group = VGroup(chord, chord_label).arrange(UP)
        
        self.place_at_grid(chord_group, 'D4', scale_factor=0.8)
        self.play(TransformFromCopy(bass, chord), TransformFromCopy(treble, chord), FadeIn(chord_label), library_icon.animate.set_color("#FF33FF"))

        # === Animation for Lecture Line 5 ===
        # Abstraction simplifies complex sound analysis.
        self.lecture[4].set_color("#FF33FF")
        self.wait(1)
